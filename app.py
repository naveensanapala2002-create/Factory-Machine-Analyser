import streamlit as st
import os
import pandas as pd
import plotly.express as px
import sqlite3
import zipfile
import re
import math
import traceback
import numpy as np
from pathlib import Path
from io import BytesIO
from datetime import datetime, time
# SQLAlchemy / psycopg2 are required only for Supabase/PostgreSQL cloud storage.
# Keep this import safe so the Streamlit app does not crash if requirements.txt was not updated.
try:
    from sqlalchemy import create_engine, text as sa_text, inspect
    SQLALCHEMY_AVAILABLE = True
    SQLALCHEMY_IMPORT_ERROR = None
except Exception as _sqlalchemy_import_error:
    create_engine = None
    sa_text = None
    inspect = None
    SQLALCHEMY_AVAILABLE = False
    SQLALCHEMY_IMPORT_ERROR = _sqlalchemy_import_error
from urllib.parse import urlparse, urlunparse, parse_qsl, urlencode, quote, unquote

# =========================================================
# APP SETTINGS
# =========================================================

st.set_page_config(
    page_title="Factory Machine Analyser",
    layout="wide"
)

# Sidebar width adjustment so dropdown and dates are clearly visible
st.markdown("""
<style>
    [data-testid="stSidebar"] {
        min-width: 370px;
        max-width: 370px;
    }
    [data-testid="stSidebarContent"] {
        padding-left: 1.1rem;
        padding-right: 1.1rem;
    }
    [data-testid="stSidebar"] label,
    [data-testid="stSidebar"] p {
        white-space: normal !important;
        line-height: 1.25rem;
    }
    [data-testid="stSidebar"] input {
        min-width: 100%;
    }
    .block-container {
        padding-top: 2rem;
    }
</style>
""", unsafe_allow_html=True)

BASE_DIR = Path(os.environ.get("FACTORY_ANALYSER_DATA_DIR", "."))
DB_DIR = BASE_DIR / "database"
EXPORT_DIR = BASE_DIR / "exports"
DB_PATH = DB_DIR / "ifactory_machine_data.db"

# RPM rule for this dashboard:
# Use only real screw RPM columns. Do not use bi_color_rpm for zone RPM analysis.
VALID_SCREW_RPM_COLUMNS = {"screw_rpm_1", "screw_rpm_2", "screw_rpm_3"}

DB_DIR.mkdir(parents=True, exist_ok=True)
EXPORT_DIR.mkdir(parents=True, exist_ok=True)

# =========================================================
# MACHINE NAME LOGIC
# =========================================================

def make_machine_key(name):
    name = str(name).lower().strip()
    key = re.sub(r"[^a-z0-9]+", "", name)
    return key


def extract_machine_name(file_name):
    name = Path(file_name).name
    name = re.sub(r"\.csv$", "", name, flags=re.IGNORECASE)

    name = re.sub(r"^machine[-_\s]*", "", name, flags=re.IGNORECASE)
    name = re.sub(r"^report[-_\s]*", "", name, flags=re.IGNORECASE)
    name = re.sub(r"^plc[-_\s]*", "", name, flags=re.IGNORECASE)
    name = re.sub(r"^ifactory[-_\s]*", "", name, flags=re.IGNORECASE)

    # Remove common date formats from the end of the file name
    name = re.sub(r"[-_\s]*\d{1,2}[-_\s]\d{1,2}[-_\s]\d{2,4}$", "", name)
    name = re.sub(r"[-_\s]*\d{4}[-_\s]\d{1,2}[-_\s]\d{1,2}$", "", name)
    name = re.sub(r"[-_\s]+$", "", name)

    return name.strip()


def is_machine_match(saved_name, imported_name):
    saved_key = make_machine_key(saved_name)
    imported_key = make_machine_key(imported_name)

    if not saved_key or not imported_key:
        return False

    if saved_key == imported_key:
        return True

    if len(saved_key) >= 6 and saved_key in imported_key:
        return True

    if len(imported_key) >= 6 and imported_key in saved_key:
        return True

    return False


def get_matching_machine_names(saved_machine_list, available_machines):
    matched_available_machines = []
    match_rows = []
    unmatched_saved = []

    for saved_machine in saved_machine_list:
        matched_names = []

        for available_machine in available_machines:
            if is_machine_match(saved_machine, available_machine):
                matched_names.append(available_machine)

        matched_names = sorted(list(set(matched_names)))

        if matched_names:
            for matched in matched_names:
                matched_available_machines.append(matched)
                match_rows.append({
                    "Saved Machine Name": saved_machine,
                    "Imported Machine Name": matched,
                    "Saved Key": make_machine_key(saved_machine),
                    "Imported Key": make_machine_key(matched),
                    "Status": "Matched"
                })
        else:
            unmatched_saved.append(saved_machine)
            match_rows.append({
                "Saved Machine Name": saved_machine,
                "Imported Machine Name": "",
                "Saved Key": make_machine_key(saved_machine),
                "Imported Key": "",
                "Status": "Not Found / No Valid Data Imported"
            })

    matched_available_machines = sorted(list(set(matched_available_machines)))
    return matched_available_machines, pd.DataFrame(match_rows), unmatched_saved


# =========================================================
# DATABASE
# =========================================================

def get_persistent_database_url():
    """
    Persistent cloud storage logic.

    Local SQLite is fine for local/factory PC use.
    For Streamlit Community Cloud, local SQLite/files are temporary and may be lost
    when the app sleeps, reboots, or redeploys.

    To make data permanent in cloud, add DATABASE_URL in Streamlit Cloud secrets:
    DATABASE_URL = "postgresql://USER:PASSWORD@HOST:PORT/DATABASE"

    Supported secret names:
    1) DATABASE_URL
    2) [database]
       url = "postgresql://..."
    3) [connections.factory_db]
       url = "postgresql://..."
    """
    url = os.environ.get("DATABASE_URL", "")

    try:
        if not url and "DATABASE_URL" in st.secrets:
            url = st.secrets["DATABASE_URL"]
    except Exception:
        pass

    try:
        if not url and "database" in st.secrets and "url" in st.secrets["database"]:
            url = st.secrets["database"]["url"]
    except Exception:
        pass

    try:
        if (
            not url
            and "connections" in st.secrets
            and "factory_db" in st.secrets["connections"]
            and "url" in st.secrets["connections"]["factory_db"]
        ):
            url = st.secrets["connections"]["factory_db"]["url"]
    except Exception:
        pass

    return str(url).strip()


def remove_psycopg2_unsupported_query_options(url):
    """
    psycopg2 does not accept Supabase's Prisma/ORM query option:
        ?pgbouncer=true

    If pgbouncer=true is present in DATABASE_URL, psycopg2 throws:
        invalid dsn: invalid connection option "pgbouncer"

    This function removes only unsupported query parameters and keeps the
    connection URL usable for SQLAlchemy + psycopg2.
    """
    if not url:
        return url

    try:
        parsed = urlparse(url)
        query_items = parse_qsl(parsed.query, keep_blank_values=True)

        # Remove query options that psycopg2 does not understand.
        allowed_items = [
            (key, value)
            for key, value in query_items
            if key.lower() not in {"pgbouncer"}
        ]

        cleaned_query = urlencode(allowed_items)

        return urlunparse((
            parsed.scheme,
            parsed.netloc,
            parsed.path,
            parsed.params,
            cleaned_query,
            parsed.fragment
        ))

    except Exception:
        return url


def convert_supabase_direct_url_to_pooler(url):
    """
    Streamlit Community Cloud can fail when connecting to Supabase direct DB host
    db.<project-ref>.supabase.co:5432 because that endpoint may resolve to IPv6.

    This function converts:
        postgresql://postgres:<password>@db.<ref>.supabase.co:5432/postgres

    into Supabase pooler format:
        postgresql://postgres.<ref>:<password>@aws-1-ap-south-1.pooler.supabase.com:6543/postgres

    Important:
    Do NOT add ?pgbouncer=true for psycopg2.
    That query option is shown by Supabase for some ORM tools, but psycopg2 rejects it.
    """
    if not url:
        return url

    try:
        parsed = urlparse(url)
        host = parsed.hostname or ""

        # If it is already a pooler URL, only clean unsupported query options.
        if "pooler.supabase.com" in host:
            return remove_psycopg2_unsupported_query_options(url)

        # Convert only Supabase direct DB host.
        if not (host.startswith("db.") and host.endswith(".supabase.co")):
            return remove_psycopg2_unsupported_query_options(url)

        project_ref = host.replace("db.", "", 1).replace(".supabase.co", "")
        username = parsed.username or "postgres"
        password = parsed.password or ""

        # Safely handle URL-encoded passwords.
        # Example: Naveen!@#143 should be stored in Streamlit Secrets as Naveen%21%40%23143.
        # When rebuilding the pooler URL, keep username/password safely encoded so @/# do not break the URL.
        username = quote(unquote(username), safe=".")
        password = quote(unquote(password), safe="") if password else ""

        # Supabase pooler username format is normally postgres.<project-ref>
        if username == "postgres":
            username = f"postgres.{project_ref}"

        pooler_host = os.environ.get("SUPABASE_POOLER_HOST", "")

        try:
            if not pooler_host and "SUPABASE_POOLER_HOST" in st.secrets:
                pooler_host = st.secrets["SUPABASE_POOLER_HOST"]
        except Exception:
            pass

        # Your Supabase project is in South Asia / Mumbai.
        # If your Supabase screen shows a different pooler host, put it in Streamlit Secrets as:
        # SUPABASE_POOLER_HOST = "your-pooler-host"
        if not pooler_host:
            pooler_host = "aws-1-ap-south-1.pooler.supabase.com"

        # Keep password exactly as supplied in the URL.
        # If password contains special characters, it should already be URL-encoded in Streamlit Secrets.
        if password:
            netloc = f"{username}:{password}@{pooler_host}:6543"
        else:
            netloc = f"{username}@{pooler_host}:6543"

        return urlunparse((
            parsed.scheme,
            netloc,
            parsed.path or "/postgres",
            "",
            "",
            ""
        ))

    except Exception:
        return remove_psycopg2_unsupported_query_options(url)


def normalize_database_url(url):
    """
    Makes DATABASE_URL suitable for SQLAlchemy + psycopg2.

    Fixes both Supabase deployment issues:
    1. Direct URL IPv6 connection issue on Streamlit Cloud.
    2. psycopg2 invalid DSN error caused by ?pgbouncer=true.
    """
    url = str(url).strip()

    if not url:
        return url

    url = convert_supabase_direct_url_to_pooler(url)
    url = remove_psycopg2_unsupported_query_options(url)

    if url.startswith("postgres://"):
        return url.replace("postgres://", "postgresql+psycopg2://", 1)

    if url.startswith("postgresql://") and "+psycopg2" not in url:
        return url.replace("postgresql://", "postgresql+psycopg2://", 1)

    return url


RAW_DATABASE_URL = get_persistent_database_url()
DATABASE_URL = normalize_database_url(RAW_DATABASE_URL)
USE_EXTERNAL_DATABASE = bool(DATABASE_URL)


def set_database_error(error):
    st.session_state["DATABASE_LAST_ERROR"] = str(error)


def clear_database_error():
    st.session_state.pop("DATABASE_LAST_ERROR", None)


def show_database_error_if_any():
    error = st.session_state.get("DATABASE_LAST_ERROR")
    if error:
        st.error("Database connection/storage error. The app is still running, but database actions are currently failing.")
        with st.expander("Show technical database error"):
            st.code(error)
        st.info("Check Streamlit Secrets DATABASE_URL, Supabase password, and Supabase project status. After correction, reboot the app.")


@st.cache_resource(show_spinner=False)
def get_external_engine():
    if not DATABASE_URL:
        return None

    if not SQLALCHEMY_AVAILABLE:
        set_database_error(
            "SQLAlchemy / psycopg2 is not installed. Please update requirements.txt with: "
            "SQLAlchemy and psycopg2-binary, then commit and reboot the app. "
            f"Import error: {SQLALCHEMY_IMPORT_ERROR}"
        )
        return None

    try:
        return create_engine(
            DATABASE_URL,
            pool_pre_ping=True,
            pool_recycle=1800,
            connect_args={"sslmode": "require", "connect_timeout": 15}
        )
    except Exception as e:
        set_database_error(e)
        return None


def get_connection():
    """
    Returns:
    - SQLAlchemy engine when DATABASE_URL is configured.
    - SQLite connection for local fallback.
    """
    if USE_EXTERNAL_DATABASE:
        return get_external_engine()

    conn = sqlite3.connect(
        DB_PATH,
        timeout=30,
        check_same_thread=False
    )
    conn.execute("PRAGMA journal_mode=WAL;")
    conn.execute("PRAGMA busy_timeout=30000;")
    return conn


def get_table_columns(table_name):
    if USE_EXTERNAL_DATABASE:
        engine = get_external_engine()
        if engine is None or inspect is None:
            return []
        inspector = inspect(engine)

        if not inspector.has_table(table_name):
            return []

        return [col["name"] for col in inspector.get_columns(table_name)]

    conn = get_connection()
    cur = conn.cursor()
    cur.execute(f"PRAGMA table_info({table_name})")
    rows = cur.fetchall()
    conn.close()
    return [row[1] for row in rows]


def create_database():
    """Create required tables. This function is safe to call repeatedly.

    Important Streamlit Cloud fix:
    Database creation is attempted lazily and errors are captured instead of crashing
    the whole app at startup.
    """
    try:
        if USE_EXTERNAL_DATABASE:
            engine = get_external_engine()
            if engine is None:
                set_database_error("External database engine is not available. Check requirements.txt, DATABASE_URL secret, and Supabase status.")
                return False

            with engine.begin() as conn:
                conn.execute(sa_text("""
                    CREATE TABLE IF NOT EXISTS machine_data (
                        id SERIAL PRIMARY KEY,
                        timestamp TEXT,
                        machine_name TEXT,
                        machine_key TEXT,
                        diameter DOUBLE PRECISION,
                        rpm DOUBLE PRECISION,
                        line_speed DOUBLE PRECISION,
                        quantity DOUBLE PRECISION,
                        diameter_source_column TEXT,
                        rpm_source_column TEXT,
                        speed_source_column TEXT,
                        data_pair TEXT,
                        source_file TEXT,
                        uploaded_date TEXT
                    )
                """))

                conn.execute(sa_text("""
                    CREATE TABLE IF NOT EXISTS saved_machines (
                        id SERIAL PRIMARY KEY,
                        machine_name TEXT UNIQUE,
                        machine_key TEXT,
                        saved_date TEXT
                    )
                """))

                required_machine_data_cols = {
                    "machine_key": "TEXT",
                    "diameter_source_column": "TEXT",
                    "rpm_source_column": "TEXT",
                    "speed_source_column": "TEXT",
                    "data_pair": "TEXT",
                    "line_speed": "DOUBLE PRECISION",
                    "quantity": "DOUBLE PRECISION",
                    "source_file": "TEXT",
                    "uploaded_date": "TEXT"
                }

                for col, dtype in required_machine_data_cols.items():
                    conn.execute(sa_text(
                        f"ALTER TABLE machine_data ADD COLUMN IF NOT EXISTS {col} {dtype}"
                    ))

                required_saved_cols = {
                    "machine_key": "TEXT",
                    "saved_date": "TEXT"
                }

                for col, dtype in required_saved_cols.items():
                    conn.execute(sa_text(
                        f"ALTER TABLE saved_machines ADD COLUMN IF NOT EXISTS {col} {dtype}"
                    ))

            clear_database_error()
            return True

        conn = get_connection()
        cur = conn.cursor()

        cur.execute("""
            CREATE TABLE IF NOT EXISTS machine_data (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT,
                machine_name TEXT,
                machine_key TEXT,
                diameter REAL,
                rpm REAL,
                line_speed REAL,
                quantity REAL,
                diameter_source_column TEXT,
                rpm_source_column TEXT,
                speed_source_column TEXT,
                source_file TEXT,
                uploaded_date TEXT
            )
        """)

        cur.execute("""
            CREATE TABLE IF NOT EXISTS saved_machines (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                machine_name TEXT UNIQUE,
                machine_key TEXT,
                saved_date TEXT
            )
        """)

        machine_data_cols = get_table_columns("machine_data")

        required_machine_data_cols = {
            "machine_key": "TEXT",
            "diameter_source_column": "TEXT",
            "rpm_source_column": "TEXT",
            "speed_source_column": "TEXT",
            "data_pair": "TEXT"
        }

        for col, dtype in required_machine_data_cols.items():
            if col not in machine_data_cols:
                cur.execute(f"ALTER TABLE machine_data ADD COLUMN {col} {dtype}")

        saved_cols = get_table_columns("saved_machines")

        required_saved_cols = {
            "machine_key": "TEXT",
            "saved_date": "TEXT"
        }

        for col, dtype in required_saved_cols.items():
            if col not in saved_cols:
                cur.execute(f"ALTER TABLE saved_machines ADD COLUMN {col} {dtype}")

        conn.commit()
        conn.close()
        clear_database_error()
        return True

    except Exception as e:
        set_database_error(e)
        return False

def save_to_database(df):
    if not create_database():
        st.error("Data was not saved because database connection is not available.")
        show_database_error_if_any()
        return False

    try:
        save_cols = [
            "timestamp",
            "machine_name",
            "machine_key",
            "diameter",
            "rpm",
            "line_speed",
            "quantity",
            "diameter_source_column",
            "rpm_source_column",
            "speed_source_column",
            "data_pair",
            "source_file",
            "uploaded_date"
        ]

        for col in save_cols:
            if col not in df.columns:
                df[col] = None

        save_df = df[save_cols].copy()

        conn = get_connection()
        if conn is None:
            set_database_error("Database connection is not available while saving data.")
            return False

        # Permanent fix for error: "too many SQL variables".
        # SQLite has a limit on how many values can be inserted in one SQL statement.
        # Earlier chunksize=5000 with method="multi" can create too many placeholders
        # when many CSV/ZIP rows are imported at once.
        #
        # Local PC / SQLite: use pandas default executemany with small chunks.
        # Supabase/PostgreSQL: use safe multi insert chunks.
        if USE_EXTERNAL_DATABASE:
            safe_chunksize = 500
            safe_method = "multi"
        else:
            safe_chunksize = 200
            safe_method = None

        save_df.to_sql(
            "machine_data",
            conn,
            if_exists="append",
            index=False,
            chunksize=safe_chunksize,
            method=safe_method
        )

        if not USE_EXTERNAL_DATABASE:
            conn.close()

        clear_database_error()
        return True

    except Exception as e:
        set_database_error(e)
        st.error("Data import failed while saving to database.")
        show_database_error_if_any()
        return False

def load_database():
    if not create_database():
        return pd.DataFrame()

    try:
        conn = get_connection()
        if conn is None:
            set_database_error("Database connection is not available while loading data.")
            return pd.DataFrame()
        df = pd.read_sql_query("SELECT * FROM machine_data", conn)

        if not USE_EXTERNAL_DATABASE:
            conn.close()

        if df.empty:
            clear_database_error()
            return df

        if "data_pair" not in df.columns:
            df["data_pair"] = ""

        df["machine_key"] = df["machine_name"].apply(make_machine_key)
        df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
        df["diameter"] = pd.to_numeric(df["diameter"], errors="coerce")
        df["rpm"] = pd.to_numeric(df["rpm"], errors="coerce")
        df["line_speed"] = pd.to_numeric(df["line_speed"], errors="coerce")
        df["quantity"] = pd.to_numeric(df["quantity"], errors="coerce")

        # Important correction:
        # If the database contains old rows imported by earlier versions where bi_color_rpm
        # was used as RPM, exclude those rows from this dashboard.
        # Current logic allows only screw_rpm_1, screw_rpm_2, screw_rpm_3 as RPM sources.
        if "rpm_source_column" in df.columns:
            df["rpm_source_clean"] = df["rpm_source_column"].apply(clean_column_name)
            df = df[df["rpm_source_clean"].isin(VALID_SCREW_RPM_COLUMNS)].copy()
            df = df.drop(columns=["rpm_source_clean"], errors="ignore")

        df = df.dropna(subset=["timestamp", "diameter", "rpm"])
        clear_database_error()
        return df

    except Exception as e:
        set_database_error(e)
        return pd.DataFrame()

def clear_database():
    if not create_database():
        st.error("Could not clear data because database connection is not available.")
        show_database_error_if_any()
        return False

    try:
        if USE_EXTERNAL_DATABASE:
            engine = get_external_engine()
            if engine is None:
                set_database_error("Database connection is not available while clearing machine data.")
                return False
            with engine.begin() as conn:
                conn.execute(sa_text("DELETE FROM machine_data"))
            clear_database_error()
            return True

        conn = get_connection()
        cur = conn.cursor()
        cur.execute("DELETE FROM machine_data")
        conn.commit()
        conn.close()
        clear_database_error()
        return True

    except Exception as e:
        set_database_error(e)
        st.error("Could not clear database.")
        show_database_error_if_any()
        return False

def save_machine_list(machine_names):
    if not create_database():
        st.error("Machine list was not saved because database connection is not available.")
        show_database_error_if_any()
        return False

    try:
        if USE_EXTERNAL_DATABASE:
            engine = get_external_engine()
            if engine is None:
                set_database_error("Database connection is not available while saving machine list.")
                return False

            with engine.begin() as conn:
                conn.execute(sa_text("DELETE FROM saved_machines"))

                for machine in machine_names:
                    machine = machine.strip()

                    if machine:
                        conn.execute(sa_text("""
                            INSERT INTO saved_machines
                            (machine_name, machine_key, saved_date)
                            VALUES (:machine_name, :machine_key, :saved_date)
                            ON CONFLICT (machine_name) DO NOTHING
                        """), {
                            "machine_name": machine,
                            "machine_key": make_machine_key(machine),
                            "saved_date": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                        })

            clear_database_error()
            return True

        conn = get_connection()
        cur = conn.cursor()

        cur.execute("BEGIN IMMEDIATE")
        cur.execute("DELETE FROM saved_machines")

        for machine in machine_names:
            machine = machine.strip()

            if machine:
                cur.execute("""
                    INSERT OR IGNORE INTO saved_machines
                    (machine_name, machine_key, saved_date)
                    VALUES (?, ?, ?)
                """, (
                    machine,
                    make_machine_key(machine),
                    datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                ))

        conn.commit()
        conn.close()
        clear_database_error()
        return True

    except Exception as e:
        set_database_error(e)
        st.error("Machine list was not saved.")
        show_database_error_if_any()
        return False

def load_saved_machine_list():
    if not create_database():
        return []

    try:
        conn = get_connection()
        if conn is None:
            set_database_error("Database connection is not available while loading saved machine list.")
            return []
        df = pd.read_sql_query(
            "SELECT machine_name FROM saved_machines ORDER BY machine_name",
            conn
        )

        if not USE_EXTERNAL_DATABASE:
            conn.close()

        if df.empty:
            clear_database_error()
            return []

        clear_database_error()
        return df["machine_name"].dropna().tolist()

    except Exception as e:
        set_database_error(e)
        return []

def clear_saved_machine_list():
    if not create_database():
        st.error("Saved machine list was not cleared because database connection is not available.")
        show_database_error_if_any()
        return False

    try:
        if USE_EXTERNAL_DATABASE:
            engine = get_external_engine()
            if engine is None:
                set_database_error("Database connection is not available while clearing saved machine list.")
                return False
            with engine.begin() as conn:
                conn.execute(sa_text("DELETE FROM saved_machines"))
            clear_database_error()
            return True

        conn = get_connection()
        cur = conn.cursor()
        cur.execute("BEGIN IMMEDIATE")
        cur.execute("DELETE FROM saved_machines")
        conn.commit()
        conn.close()
        clear_database_error()
        return True

    except Exception as e:
        set_database_error(e)
        st.error("Saved machine list was not cleared.")
        show_database_error_if_any()
        return False



# Database is created lazily inside database functions.
# This prevents the deployed Streamlit app from crashing at startup if the external database is temporarily unavailable.

# =========================================================
# COLUMN LOGIC
# =========================================================

def clean_column_name(col):
    col = str(col).strip().lower()
    col = col.replace(" ", "_")
    col = col.replace("-", "_")
    col = col.replace(".", "_")
    col = col.replace("/", "_")
    col = re.sub(r"_+", "_", col)
    return col


def get_column_lookup(df):
    return {clean_column_name(col): col for col in df.columns}


def is_valid_screw_rpm_source(source_column):
    """
    True only for screw_rpm_1, screw_rpm_2, screw_rpm_3.
    This prevents old database rows imported with bi_color_rpm from being used as Screw RPM.
    """
    return clean_column_name(source_column) in VALID_SCREW_RPM_COLUMNS


def find_text_column(df, possible_names):
    lookup = get_column_lookup(df)

    for name in possible_names:
        cleaned = clean_column_name(name)

        if cleaned in lookup:
            return lookup[cleaned]

    return None


def get_existing_columns_in_order(df, ordered_names):
    lookup = get_column_lookup(df)
    found_cols = []

    for name in ordered_names:
        cleaned = clean_column_name(name)
        if cleaned in lookup:
            found_cols.append(lookup[cleaned])

    return found_cols


def find_best_numeric_column(df, possible_names):
    lookup = get_column_lookup(df)
    candidates = []

    for name in possible_names:
        cleaned = clean_column_name(name)

        if cleaned in lookup:
            actual_col = lookup[cleaned]
            numeric_series = pd.to_numeric(df[actual_col], errors="coerce")
            valid_count = numeric_series.notna().sum()
            non_zero_count = (numeric_series.fillna(0) != 0).sum()

            candidates.append({
                "column": actual_col,
                "valid_count": valid_count,
                "non_zero_count": non_zero_count
            })

    if not candidates:
        return None

    candidates_df = pd.DataFrame(candidates)
    candidates_df = candidates_df.sort_values(
        by=["non_zero_count", "valid_count"],
        ascending=False
    )

    if candidates_df.iloc[0]["valid_count"] == 0:
        return None

    return candidates_df.iloc[0]["column"]


def convert_decimal_to_single_value(series):
    """
    Converts decimal machine readings into a safe single integer value.

    Example:
    2.28800010681152 -> 2

    Important:
    Pandas cannot directly convert 2.288 to Int64 because it is not an exact integer.
    So decimal values are truncated first, then converted to nullable Int64.
    """
    numeric = pd.to_numeric(series, errors="coerce")
    truncated = np.trunc(numeric)
    return pd.Series(truncated, index=series.index).astype("Int64")


def column_has_valid_numeric_data(df, col):
    """Return True only when the column exists and has at least one valid non-blank numeric reading."""
    if col not in df.columns:
        return False

    numeric = pd.to_numeric(df[col], errors="coerce")
    return numeric.notna().sum() > 0


def get_valid_columns_in_order(df, preferred_columns):
    """Keep preferred order, but remove columns that are missing or fully blank/non-numeric."""
    lookup = get_column_lookup(df)
    valid_cols = []

    for col in preferred_columns:
        cleaned = clean_column_name(col)
        if cleaned in lookup:
            actual_col = lookup[cleaned]
            if column_has_valid_numeric_data(df, actual_col):
                valid_cols.append(actual_col)

    return valid_cols


def select_diameter_columns(df):
    """
    Priority logic with valid-data check:
    1. First try diameter_1 and diameter_2.
    2. If these columns are missing OR fully blank, use diameter_hot_1 and diameter_hot_2.

    This fixes the issue where a file contains diameter_1/diameter_2 column names
    but the actual data is present only in diameter_hot_1/diameter_hot_2.
    """
    diameter_cols = get_valid_columns_in_order(df, ["diameter_1", "diameter_2"])

    if diameter_cols:
        return diameter_cols, "diameter_1/diameter_2 with valid data"

    hot_diameter_cols = get_valid_columns_in_order(df, ["diameter_hot_1", "diameter_hot_2"])

    if hot_diameter_cols:
        return hot_diameter_cols, "diameter_hot_1/diameter_hot_2 fallback because diameter_1/diameter_2 were missing or blank"

    return [], "No valid diameter column found"


def select_rpm_columns(df):
    """
    Screw RPM columns are taken in fixed order and only valid numeric columns are used:
    screw_rpm_1, screw_rpm_2, screw_rpm_3.

    bi_color_rpm is intentionally NOT used for Screw RPM analysis.
    """
    return get_valid_columns_in_order(df, ["screw_rpm_1", "screw_rpm_2", "screw_rpm_3"])


def standardize_csv(df, file_name):
    original_columns = list(df.columns)

    timestamp_col = find_text_column(df, [
        "timestamp",
        "time",
        "date_time",
        "datetime",
        "created_at",
        "recorded_at"
    ])

    diameter_cols, diameter_selection_logic = select_diameter_columns(df)
    rpm_cols = select_rpm_columns(df)

    # Speed requirement:
    # Use only the exact "speed" column for speed analysis.
    # Do not automatically take traction_speed, line_speed, bi_color_speed, etc.
    speed_col = find_best_numeric_column(df, [
        "speed"
    ])

    quantity_col = find_best_numeric_column(df, [
        "quantity",
        "qty",
        "production",
        "meter",
        "meters",
        "length"
    ])

    missing = []

    if timestamp_col is None:
        missing.append("timestamp/time")

    if not diameter_cols:
        missing.append("diameter_1/diameter_2 or diameter_hot_1/diameter_hot_2")

    if not rpm_cols:
        missing.append("screw_rpm_1/screw_rpm_2/screw_rpm_3")

    if missing:
        return None, missing, original_columns, {}

    machine_name = extract_machine_name(file_name)
    output_parts = []

    for dia_index, diameter_col in enumerate(diameter_cols):
        # Pair diameter columns with RPM columns in order.
        # Example: diameter_1 -> screw_rpm_1, diameter_2 -> screw_rpm_2.
        # If a matching RPM is not available, the first RPM column is used.
        if dia_index < len(rpm_cols):
            rpm_col = rpm_cols[dia_index]
        else:
            rpm_col = rpm_cols[0]

        part = pd.DataFrame()
        part["timestamp"] = pd.to_datetime(df[timestamp_col], errors="coerce")
        part["machine_name"] = machine_name
        part["machine_key"] = make_machine_key(machine_name)

        # Convert decimal readings into single integer values.
        part["diameter"] = convert_decimal_to_single_value(df[diameter_col])
        part["rpm"] = convert_decimal_to_single_value(df[rpm_col])

        if speed_col:
            part["line_speed"] = pd.to_numeric(df[speed_col], errors="coerce")
        else:
            part["line_speed"] = 0

        if quantity_col:
            part["quantity"] = pd.to_numeric(df[quantity_col], errors="coerce")
        else:
            part["quantity"] = 0

        part["diameter_source_column"] = diameter_col
        part["rpm_source_column"] = rpm_col
        part["speed_source_column"] = speed_col
        part["data_pair"] = f"{diameter_col} + {rpm_col}"
        part["source_file"] = file_name
        part["uploaded_date"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        part = part.dropna(subset=["timestamp", "diameter", "rpm"])
        output_parts.append(part)

    if not output_parts:
        return None, ["no valid timestamp/diameter/rpm rows after cleaning"], original_columns, {}

    new_df = pd.concat(output_parts, ignore_index=True)

    # Remove invalid or zero diameter rows if present
    new_df = new_df[pd.to_numeric(new_df["diameter"], errors="coerce").notna()]
    new_df = new_df[pd.to_numeric(new_df["rpm"], errors="coerce").notna()]

    column_info = {
        "timestamp_column": timestamp_col,
        "diameter_selection_logic": diameter_selection_logic,
        "diameter_columns_used": ", ".join(diameter_cols),
        "rpm_columns_used": ", ".join(rpm_cols),
        "speed_column": speed_col,
        "quantity_column": quantity_col,
        "rows_after_cleaning": len(new_df)
    }

    if new_df.empty:
        return None, ["no valid timestamp/diameter/rpm rows after cleaning"], original_columns, column_info

    return new_df, [], original_columns, column_info


# =========================================================
# FILE PROCESSING
# =========================================================

def is_no_data_file(df):
    if df.empty:
        return True

    first_text = str(df.iloc[0].to_string()).lower()

    if "no data available" in first_text:
        return True

    return False


def read_csv_from_file_object(file_obj, file_name):
    try:
        df = pd.read_csv(file_obj)
    except Exception:
        try:
            file_obj.seek(0)
        except Exception:
            pass

        df = pd.read_csv(file_obj, encoding="latin1")

    if df.empty:
        return None, ["empty file"], [], {}

    if is_no_data_file(df):
        return None, ["no data available"], list(df.columns), {}

    return standardize_csv(df, file_name)


def process_uploaded_files(uploaded_files, allowed_machine_names=None):
    """
    Imports only files that match the Current Frozen Machine List.

    Requirement:
    - When CSV/ZIP is uploaded, process only CSV files whose extracted machine name
      matches one of the saved/frozen machines.
    - Skip all other machines before reading/importing, so unwanted machine data
      does not occupy database storage.
    """
    imported_data = []
    error_list = []
    import_log = []

    allowed_machine_names = allowed_machine_names or []
    allowed_machine_names = [str(m).strip() for m in allowed_machine_names if str(m).strip()]

    if not allowed_machine_names:
        error_list.append({
            "file": "All uploaded files",
            "issue": "No Current Frozen Machine List found. Please save/freeze machine names first in Machine List Setup.",
            "columns_found": ""
        })
        return pd.DataFrame(), error_list, import_log

    def is_allowed_machine_file(candidate_file_name):
        imported_machine_name = extract_machine_name(candidate_file_name)

        for saved_machine in allowed_machine_names:
            if is_machine_match(saved_machine, imported_machine_name):
                return True

        return False

    def add_skipped_log(candidate_file_name):
        imported_machine_name = extract_machine_name(candidate_file_name)

        import_log.append({
            "file": candidate_file_name,
            "machine_name": imported_machine_name,
            "machine_key": make_machine_key(imported_machine_name),
            "status": "Skipped - Not in Current Frozen Machine List",
            "rows": 0
        })

    for uploaded_file in uploaded_files:
        file_name = uploaded_file.name

        if file_name.lower().endswith(".csv"):
            if not is_allowed_machine_file(file_name):
                add_skipped_log(file_name)
                continue

            result_df, missing, columns, info = read_csv_from_file_object(uploaded_file, file_name)

            if result_df is not None and not result_df.empty:
                imported_data.append(result_df)

                import_log.append({
                    "file": file_name,
                    "machine_name": extract_machine_name(file_name),
                    "machine_key": make_machine_key(extract_machine_name(file_name)),
                    "status": "Imported",
                    "rows": len(result_df),
                    **info
                })
            else:
                error_list.append({
                    "file": file_name,
                    "issue": ", ".join(missing),
                    "columns_found": ", ".join(map(str, columns))
                })

        elif file_name.lower().endswith(".zip"):
            try:
                zip_bytes = BytesIO(uploaded_file.read())

                with zipfile.ZipFile(zip_bytes, "r") as zip_ref:
                    for inner_file in zip_ref.namelist():
                        if inner_file.lower().endswith(".csv"):
                            csv_name = Path(inner_file).name

                            if not is_allowed_machine_file(csv_name):
                                add_skipped_log(csv_name)
                                continue

                            with zip_ref.open(inner_file) as csv_file:
                                result_df, missing, columns, info = read_csv_from_file_object(csv_file, csv_name)

                                if result_df is not None and not result_df.empty:
                                    imported_data.append(result_df)

                                    import_log.append({
                                        "file": csv_name,
                                        "machine_name": extract_machine_name(csv_name),
                                        "machine_key": make_machine_key(extract_machine_name(csv_name)),
                                        "status": "Imported",
                                        "rows": len(result_df),
                                        **info
                                    })
                                else:
                                    error_list.append({
                                        "file": csv_name,
                                        "issue": ", ".join(missing),
                                        "columns_found": ", ".join(map(str, columns))
                                    })

            except Exception as e:
                error_list.append({
                    "file": file_name,
                    "issue": str(e),
                    "columns_found": ""
                })

    if imported_data:
        final_df = pd.concat(imported_data, ignore_index=True)
    else:
        final_df = pd.DataFrame()

    return final_df, error_list, import_log

# =========================================================
# MACHINE LIST PARSING
# =========================================================

def parse_machine_text(text):
    if not text:
        return []

    parts = re.split(r"[\n,;\t]+", text)
    machines = []

    for part in parts:
        machine = part.strip()

        if machine:
            machines.append(machine)

    return sorted(list(set(machines)))


# =========================================================
# CONTINUOUS DIAMETER ZONE ANALYSIS
# =========================================================

def format_duration(minutes):
    if pd.isna(minutes):
        return ""

    total_seconds = int(round(minutes * 60))
    hours = total_seconds // 3600
    minutes_left = (total_seconds % 3600) // 60
    seconds_left = total_seconds % 60

    if hours > 0:
        return f"{hours}h {minutes_left}m {seconds_left}s"

    return f"{minutes_left}m {seconds_left}s"


def get_most_repeated_positive_value(series):
    """Return the most repeated numeric value from values greater than 0."""
    numeric = pd.to_numeric(series, errors="coerce")
    numeric = numeric[numeric > 0].dropna()

    if numeric.empty:
        return None

    counts = numeric.value_counts(dropna=True)
    max_count = counts.max()
    candidates = counts[counts == max_count].index.tolist()

    # When two values have the same frequency, choose the lower value for stable output.
    return sorted(candidates)[0]


def create_continuous_diameter_zone_analysis(df, machine_name, min_duration_minutes=30):
    """
    Final logic for dashboard zone tables, applied to every selected machine:

    1. Diameter decimal readings are converted to integer values.
       Example: 1.870 -> 1.
    2. Only rows where Diameter > 0, Screw RPM > 0 and Speed > 0 are considered.
    3. A primary zone starts when the integer Diameter changes.
    4. Inside each same-diameter primary zone, split again whenever Speed or Screw RPM changes.
    5. Every continuous Diameter + Speed + RPM segment with duration > 30 minutes is shown.

    This means every selected machine follows the same logic.
    The dashboard will not keep only one most-repeated speed.
    It will show all valid highlighted rows such as:
    Diameter 1 | Speed 40   | RPM 8 | duration > 30 minutes
    Diameter 1 | Speed 45   | RPM 9 | duration > 30 minutes
    Diameter 1 | Speed 50   | RPM 9 | duration > 30 minutes
    Diameter 1 | Speed 33.9 | RPM 6 | duration > 30 minutes
    """
    if df.empty:
        return pd.DataFrame()

    work = df.sort_values(["data_pair", "timestamp"]).copy()
    work = work.dropna(subset=["timestamp", "diameter", "rpm"])

    work["diameter"] = convert_decimal_to_single_value(work["diameter"])
    work["rpm"] = convert_decimal_to_single_value(work["rpm"])
    work["line_speed"] = pd.to_numeric(work.get("line_speed", 0), errors="coerce")

    # Requirement: Diameter, Screw RPM and Speed must be greater than 0 for all output tables.
    work = work.dropna(subset=["diameter", "rpm", "line_speed"])
    work = work[
        (pd.to_numeric(work["diameter"], errors="coerce") > 0) &
        (pd.to_numeric(work["rpm"], errors="coerce") > 0) &
        (pd.to_numeric(work["line_speed"], errors="coerce") > 0)
    ].copy()

    if work.empty:
        return pd.DataFrame()

    # Stable numeric values for segmenting and display.
    work["diameter"] = pd.to_numeric(work["diameter"], errors="coerce").astype("Int64")
    work["rpm"] = pd.to_numeric(work["rpm"], errors="coerce").astype("Int64")
    work["line_speed"] = pd.to_numeric(work["line_speed"], errors="coerce").round(3)

    all_results = []

    for data_pair, pair_df in work.groupby("data_pair", dropna=False):
        pair_df = pair_df.sort_values("timestamp").copy()

        # Primary zone starts when diameter changes.
        pair_df["diameter_zone_id"] = (pair_df["diameter"] != pair_df["diameter"].shift()).cumsum()

        for diameter_zone_id, zone_df in pair_df.groupby("diameter_zone_id"):
            zone_df = zone_df.sort_values("timestamp").copy()
            if zone_df.empty:
                continue

            # Inside the same diameter zone, every continuous Speed + RPM combination is separate.
            segment_change = (
                (zone_df["line_speed"] != zone_df["line_speed"].shift()) |
                (zone_df["rpm"] != zone_df["rpm"].shift())
            )
            zone_df["speed_rpm_segment_id"] = segment_change.cumsum()

            for segment_id, segment_df in zone_df.groupby("speed_rpm_segment_id"):
                segment_df = segment_df.sort_values("timestamp").copy()

                if segment_df.empty:
                    continue

                segment_start_time = segment_df["timestamp"].min()
                segment_end_time = segment_df["timestamp"].max()
                segment_duration_minutes = (segment_end_time - segment_start_time).total_seconds() / 60

                # User condition: every selected Diameter + Speed + RPM segment should be more than 30 minutes.
                if segment_duration_minutes <= min_duration_minutes:
                    continue

                segment_reading_count = len(segment_df)
                diameter_value = int(segment_df["diameter"].iloc[0])
                segment_speed = float(segment_df["line_speed"].iloc[0])
                segment_rpm = int(segment_df["rpm"].iloc[0])

                # Since segment is split by Speed + RPM, min/max are normally same.
                rpm_min = segment_df["rpm"].min()
                rpm_max = segment_df["rpm"].max()
                rpm_min_row = segment_df.loc[segment_df["rpm"].idxmin()]
                rpm_max_row = segment_df.loc[segment_df["rpm"].idxmax()]
                first_segment_row = segment_df.sort_values("timestamp").iloc[0]

                all_results.append({
                    "Machine": machine_name,
                    "Data Pair": data_pair,
                    "Zone No": len(all_results) + 1,
                    "Diameter": diameter_value,
                    "Screw RPM Min": int(rpm_min) if pd.notna(rpm_min) else None,
                    "Screw RPM Max": int(rpm_max) if pd.notna(rpm_max) else None,
                    # Compatibility columns kept, but now they represent every valid segment.
                    "Most Repeated Speed": round(segment_speed, 3),
                    "RPM at Most Repeated Speed": segment_rpm,
                    "Speed at RPM Max": round(segment_speed, 3),
                    # These timestamps and duration are for this exact Diameter + Speed + RPM segment.
                    "Start Timestamp": segment_start_time,
                    "End Timestamp": segment_end_time,
                    "Duration Minutes": round(segment_duration_minutes, 2),
                    "Duration": format_duration(segment_duration_minutes),
                    "Readings Count": segment_reading_count,
                    "RPM Min Timestamp": rpm_min_row["timestamp"],
                    "RPM Max Timestamp": rpm_max_row["timestamp"],
                    "Most Repeated RPM Timestamp": first_segment_row["timestamp"],
                    "Diameter Source": segment_df["diameter_source_column"].iloc[0],
                    "RPM Source": segment_df["rpm_source_column"].iloc[0],
                    "Speed Source": segment_df["speed_source_column"].iloc[0] if "speed_source_column" in segment_df.columns else None,
                    "Source File": segment_df["source_file"].iloc[0]
                })

    result = pd.DataFrame(all_results)

    if not result.empty:
        result = result.sort_values(
            ["Machine", "Start Timestamp", "Data Pair", "Zone No"],
            ascending=True
        ).reset_index(drop=True)
        result["Zone No"] = result.groupby("Machine").cumcount() + 1

    return result

def create_rpm_max_ascending_zone_table(zone_result):
    """
    Creates the additional analysis table requested:
    - RPM Max ascending order
    - Includes speed at RPM max
    - Supports one or multiple selected machines
    - Only considers Screw RPM Max values greater than 0
    """
    if zone_result.empty:
        return pd.DataFrame()

    result = zone_result.copy()

    # Requirement: Diameter and Screw RPM must be greater than 0.
    if "Diameter" in result.columns:
        result["Diameter"] = pd.to_numeric(result["Diameter"], errors="coerce")
        result = result[result["Diameter"] > 0].copy()

    # Requirement: while taking Screw RPM, consider only values greater than 0.
    # This prevents zero RPM zones from appearing in the RPM Max Ascending Zone Table.
    if "Screw RPM Max" in result.columns:
        result["Screw RPM Max"] = pd.to_numeric(result["Screw RPM Max"], errors="coerce")
        result = result[result["Screw RPM Max"] > 0].copy()

    if result.empty:
        return pd.DataFrame()

    result = result.rename(columns={
        "Speed at RPM Max": "Speed"
    })

    # Requirement: while taking Speed, consider only values greater than 0.
    # This prevents 0 speed values from appearing in the RPM Max Ascending Zone Table.
    if "Speed" in result.columns:
        result["Speed"] = pd.to_numeric(result["Speed"], errors="coerce")
        result = result[result["Speed"] > 0].copy()

    if result.empty:
        return pd.DataFrame()

    required_cols = [
        "Machine",
        "Zone No",
        "Diameter",
        "Screw RPM Max",
        "Speed",
        "Most Repeated Speed",
        "RPM at Most Repeated Speed",
        "Start Timestamp",
        "End Timestamp",
        "Duration",
        "Duration Minutes",
        "Readings Count",
        "Data Pair",
        "RPM Max Timestamp",
        "Diameter Source",
        "RPM Source",
        "Speed Source"
    ]

    for col in required_cols:
        if col not in result.columns:
            result[col] = None

    result = result[required_cols].copy()
    result = result.sort_values(
        by=["Machine", "Screw RPM Max", "Start Timestamp"],
        ascending=[True, True, True]
    ).reset_index(drop=True)

    return result



def create_same_diameter_selected_machines_table(zone_result, selected_machines):
    """
    Creates an additional comparison table below RPM Max Ascending Zone Table.

    Requirement:
    - Compare all machines selected in Zone Analysis Dropdown.
    - Find diameter values that are common across all selected machines.
    - For common diameters, show ALL valid continuous Speed + RPM segments from every selected machine.
    - Diameter, Speed and Screw RPM must all be greater than 0.
    - Each exact Diameter + Speed + RPM segment must have duration greater than 30 minutes.
    - Existing tables are not disturbed.
    """
    if zone_result.empty or not selected_machines or len(selected_machines) < 2:
        return pd.DataFrame()

    work = zone_result.copy()

    if "Machine" not in work.columns or "Diameter" not in work.columns:
        return pd.DataFrame()

    rpm_col = "RPM at Most Repeated Speed" if "RPM at Most Repeated Speed" in work.columns else "Screw RPM Max"
    speed_col = "Most Repeated Speed" if "Most Repeated Speed" in work.columns else "Speed at RPM Max"

    if rpm_col not in work.columns or speed_col not in work.columns:
        return pd.DataFrame()

    work["Diameter"] = pd.to_numeric(work["Diameter"], errors="coerce")
    work[rpm_col] = pd.to_numeric(work[rpm_col], errors="coerce")
    work[speed_col] = pd.to_numeric(work[speed_col], errors="coerce")

    # Diameter, RPM and Speed must all be greater than 0 for comparison.
    work = work[
        (work["Diameter"] > 0) &
        (work[rpm_col] > 0) &
        (work[speed_col] > 0)
    ].copy()

    if work.empty:
        return pd.DataFrame()

    selected_machine_set = set(selected_machines)

    diameter_machine_count = (
        work.groupby("Diameter")["Machine"]
        .apply(lambda s: len(set(s).intersection(selected_machine_set)))
        .reset_index(name="Selected Machines Available")
    )

    # Strict comparison: same diameter should be available in all selected machines.
    common_diameters = diameter_machine_count[
        diameter_machine_count["Selected Machines Available"] == len(selected_machine_set)
    ]["Diameter"].tolist()

    if not common_diameters:
        return pd.DataFrame()

    result = work[work["Diameter"].isin(common_diameters)].copy()

    # Output uses the priority requested by the user:
    # Speed = most repeated Speed in that same diameter zone.
    # Screw RPM = RPM linked to that most repeated Speed.
    result["Selected Machine"] = result["Machine"]
    result["Same Diameter in Selected Machines"] = result["Diameter"]
    result["Screw RPM"] = result[rpm_col]
    result["Speed"] = result[speed_col]
    result["Start Time"] = result.get("Start Timestamp")
    result["End Time"] = result.get("End Timestamp")
    result["Duration in Minutes"] = result.get("Duration Minutes")

    output_cols = [
        "Selected Machine",
        "Same Diameter in Selected Machines",
        "Screw RPM",
        "Speed",
        "Start Time",
        "End Time",
        "Duration",
        "Duration in Minutes",
        "Data Pair",
        "Most Repeated RPM Timestamp",
        "RPM Max Timestamp",
        "Diameter Source",
        "RPM Source",
        "Speed Source"
    ]

    for col in output_cols:
        if col not in result.columns:
            result[col] = None

    result = result[output_cols].copy()

    result = result.sort_values(
        by=["Same Diameter in Selected Machines", "Selected Machine", "Speed", "Screw RPM", "Start Time"],
        ascending=[True, True, True, True, True]
    ).reset_index(drop=True)

    result.insert(0, "Sl.No", range(1, len(result) + 1))

    return result

def convert_to_excel(zone_result, rpm_ascending_result, same_diameter_result, filtered_data, match_table):
    output = BytesIO()

    with pd.ExcelWriter(output, engine="openpyxl") as writer:
        zone_result.to_excel(writer, index=False, sheet_name="Continuous Diameter Zones")
        rpm_ascending_result.to_excel(writer, index=False, sheet_name="RPM Max Ascending Zones")
        same_diameter_result.to_excel(writer, index=False, sheet_name="Same Diameter Comparison")
        filtered_data.to_excel(writer, index=False, sheet_name="Selected Machines Raw Data")

        if not match_table.empty:
            match_table.to_excel(writer, index=False, sheet_name="Machine Match Status")

    output.seek(0)
    return output


def main():
    # =========================================================
    # UI
    # =========================================================

    st.title("Factory Machine Analyser")

    st.write("""
    This dashboard analyses iFactory CSV/ZIP files and creates continuous diameter zones.
    A zone is created only when the same integer diameter continues for more than 30 minutes.
    For each valid zone, the dashboard shows diameter, screw RPM minimum/maximum, timestamps, and duration.
    """)

    tab1, tab2, tab3, tab4 = st.tabs([
        "1. Upload Data",
        "2. Machine List Setup",
        "3. Zone Analysis Dashboard",
        "4. Database Control"
    ])

    # =========================================================
    # TAB 1 - UPLOAD DATA
    # =========================================================

    with tab1:
        st.header("Upload iFactory CSV or ZIP Files")

        st.info("""
        Upload your iFactory ZIP/CSV file here and click **Import Uploaded Files to Database**.

        Column selection logic:
        - Diameter: first preference **diameter_1 + diameter_2**
        - If not available: **diameter_hot_1 + diameter_hot_2**
        - Screw RPM: **screw_rpm_1, screw_rpm_2, screw_rpm_3** in order
        - **bi_color_rpm is not used as Screw RPM**
        - Decimal values are converted into single integer values, example: **2.28800010681152 → 2**
        - Import filter: only files matching **Current Frozen Machine List** are imported; remaining machines are skipped to save storage.
        """)

        uploaded_files = st.file_uploader(
            "Upload CSV or ZIP files",
            type=["csv", "zip"],
            accept_multiple_files=True
        )

        if uploaded_files:
            if st.button("Import Uploaded Files to Database"):
                saved_machine_filter = load_saved_machine_list()
                imported_df, errors, import_log = process_uploaded_files(
                    uploaded_files,
                    allowed_machine_names=saved_machine_filter
                )

                if not imported_df.empty:
                    if save_to_database(imported_df):
                        st.success(f"Imported successfully: {len(imported_df)} rows")

                    st.subheader("Import Log")
                    st.dataframe(pd.DataFrame(import_log), width="stretch")

                    st.subheader("Imported Machine Names")
                    imported_names = sorted(imported_df["machine_name"].dropna().unique())

                    st.dataframe(
                        pd.DataFrame({
                            "Imported Machine Name": imported_names,
                            "Machine Key": [make_machine_key(m) for m in imported_names]
                        }),
                        width="stretch"
                    )

                    st.subheader("Imported Data Preview")
                    st.dataframe(imported_df.head(100), width="stretch")

                else:
                    st.error("No valid data imported. Please check skipped files below.")

                if errors:
                    st.warning("Some files were skipped.")
                    st.dataframe(pd.DataFrame(errors), width="stretch")

        current_data = load_database()
        show_database_error_if_any()

        if USE_EXTERNAL_DATABASE:
            st.success("Storage Mode: Persistent cloud database connected. Uploaded data will remain after app sleep/reboot.")
        else:
            st.warning("Storage Mode: Local SQLite. On Streamlit Cloud this is temporary and may be lost after app sleep/reboot. Add DATABASE_URL in app Secrets for permanent storage.")

        st.subheader("Current Database Status")
        st.metric("Total Stored Rows", len(current_data))

    # =========================================================
    # TAB 2 - MACHINE LIST SETUP
    # =========================================================

    with tab2:
        st.header("Machine List Setup / Freeze Machines")

        st.info("""
        Paste machine names here. Same matching logic will apply to all current and future machines.
        Symbols like `/`, `-`, spaces, `_`, `+`, and brackets are ignored during matching.
        """)

        saved_machines = load_saved_machine_list()

        if saved_machines:
            st.success(f"Machine list is saved/frozen. Total saved machines: {len(saved_machines)}")

            st.subheader("Current Frozen Machine List")
            st.dataframe(
                pd.DataFrame({
                    "Machine Name": saved_machines,
                    "Machine Key": [make_machine_key(m) for m in saved_machines]
                }),
                width="stretch"
            )

            default_text = "\n".join(saved_machines)
        else:
            st.warning("No saved machine list found. Please paste machine names and save.")
            default_text = ""

        machine_text = st.text_area(
            "Paste machine names here",
            value=default_text,
            height=300,
            placeholder="Example:\nGC-SHEATHING-80D\nRC-SHEATHING-80/100A\nGC EXTRUDER-70A"
        )

        parsed_machines = parse_machine_text(machine_text)

        st.subheader("Preview of Machine Names to Save")

        if parsed_machines:
            st.dataframe(
                pd.DataFrame({
                    "Machine Name": parsed_machines,
                    "Machine Key": [make_machine_key(m) for m in parsed_machines]
                }),
                width="stretch"
            )
            st.info(f"Total machines in preview: {len(parsed_machines)}")
        else:
            st.info("Paste machine names to preview.")

        col1, col2 = st.columns(2)

        with col1:
            if st.button("Save / Freeze Machine List"):
                if parsed_machines:
                    save_machine_list(parsed_machines)
                    st.success(f"Machine list saved successfully. Total saved machines: {len(parsed_machines)}")
                    st.info("Go to Zone Analysis Dashboard tab to analyse.")
                else:
                    st.error("Please paste at least one machine name.")

        with col2:
            if st.button("Clear Saved Machine List"):
                clear_saved_machine_list()
                st.success("Saved machine list cleared.")

    # =========================================================
    # TAB 3 - DASHBOARD
    # =========================================================

    with tab3:
        st.header("Continuous Diameter Zone Analysis")

        data = load_database()
        show_database_error_if_any()

        if data.empty:
            st.warning("No data found in database. First upload CSV or ZIP files in Upload Data tab.")
            st.info("Go to **1. Upload Data**, upload your ZIP file, and click **Import Uploaded Files to Database**.")
        else:
            saved_machine_list = load_saved_machine_list()

            data = data.sort_values("timestamp").copy()
            global_min_dt = data["timestamp"].min()
            global_max_dt = data["timestamp"].max()

            st.sidebar.header("Zone Analysis Dropdown")

            # Multiselect machine selector at the top of the sidebar.
            all_available_machines = sorted(data["machine_name"].dropna().unique().tolist())

            if saved_machine_list:
                all_matched_machines, _, _ = get_matching_machine_names(
                    saved_machine_list,
                    all_available_machines
                )
                top_machine_dropdown_list = all_matched_machines
            else:
                top_machine_dropdown_list = all_available_machines

            if not top_machine_dropdown_list:
                st.warning("No machine data is available. Please upload valid CSV/ZIP data first.")
                st.stop()

            selected_machines = st.sidebar.multiselect(
                "Select Machine for Zone Analysis",
                top_machine_dropdown_list,
                default=[top_machine_dropdown_list[0]]
            )

            if not selected_machines:
                st.warning("Please select at least one machine for zone analysis.")
                st.stop()

            st.sidebar.markdown("### Select Date & Time Range")

            start_date = st.sidebar.date_input(
                "Start Date",
                value=global_min_dt.date(),
                min_value=global_min_dt.date(),
                max_value=global_max_dt.date(),
                format="YYYY/MM/DD",
                key="zone_start_date"
            )

            start_time = st.sidebar.time_input(
                "Start Time",
                value=global_min_dt.time().replace(microsecond=0),
                step=60,
                key="zone_start_time"
            )

            end_date = st.sidebar.date_input(
                "End Date",
                value=global_max_dt.date(),
                min_value=global_min_dt.date(),
                max_value=global_max_dt.date(),
                format="YYYY/MM/DD",
                key="zone_end_date"
            )

            end_time = st.sidebar.time_input(
                "End Time",
                value=global_max_dt.time().replace(microsecond=0),
                step=60,
                key="zone_end_time"
            )

            start_datetime = datetime.combine(start_date, start_time)
            end_datetime = datetime.combine(end_date, end_time)

            if start_datetime > end_datetime:
                st.sidebar.error("Start Date/Time should not be after End Date/Time.")
                st.stop()

            range_data = data[
                (data["timestamp"] >= pd.Timestamp(start_datetime)) &
                (data["timestamp"] <= pd.Timestamp(end_datetime))
            ].copy()

            available_machines_in_range = sorted(range_data["machine_name"].dropna().unique().tolist())

            match_table = pd.DataFrame()
            unmatched_saved_machines = []

            if saved_machine_list:
                _, match_table, unmatched_saved_machines = get_matching_machine_names(
                    saved_machine_list,
                    available_machines_in_range
                )

            st.sidebar.caption(
                "Selected range: "
                f"{start_datetime.strftime('%d-%m-%Y %I:%M %p')} to "
                f"{end_datetime.strftime('%d-%m-%Y %I:%M %p')}"
            )

            selected_data_parts = []
            zone_result_parts = []
            machines_without_data = []

            for machine in selected_machines:
                machine_data = range_data[
                    range_data["machine_name"].apply(
                        lambda imported_machine: is_machine_match(machine, imported_machine)
                    )
                ].copy()

                if machine_data.empty:
                    machines_without_data.append(machine)
                    continue

                machine_data = machine_data.sort_values(["data_pair", "timestamp"])
                selected_data_parts.append(machine_data)

                machine_zone_result = create_continuous_diameter_zone_analysis(
                    machine_data,
                    machine,
                    min_duration_minutes=30
                )

                if not machine_zone_result.empty:
                    zone_result_parts.append(machine_zone_result)

            if selected_data_parts:
                selected_data = pd.concat(selected_data_parts, ignore_index=True)
            else:
                selected_data = pd.DataFrame()

            if zone_result_parts:
                zone_result = pd.concat(zone_result_parts, ignore_index=True)
                zone_result = zone_result.sort_values(
                    ["Machine", "Start Timestamp", "Data Pair", "Zone No"],
                    ascending=True
                ).reset_index(drop=True)
            else:
                zone_result = pd.DataFrame()

            rpm_ascending_result = create_rpm_max_ascending_zone_table(zone_result)
            same_diameter_result = create_same_diameter_selected_machines_table(
                zone_result,
                selected_machines
            )

            st.subheader("Selected Date & Time Range")
            st.info(
                f"{start_datetime.strftime('%d-%m-%Y %I:%M %p')}  to  "
                f"{end_datetime.strftime('%d-%m-%Y %I:%M %p')}"
            )

            st.subheader("Selected Machines")
            st.success(", ".join(selected_machines))

            if machines_without_data:
                st.warning(
                    "These selected machines have no data in the selected date/time range: "
                    + ", ".join(machines_without_data)
                )

            if saved_machine_list:
                st.subheader("Machine Match Status")
                st.caption("This table updates based on the selected Start Date/Time and End Date/Time range.")
                st.dataframe(match_table, width="stretch")

                if unmatched_saved_machines:
                    st.warning(
                        "Some saved machines are not found in the selected date/time range. "
                        "If data exists outside this range, change Start Date/Time and End Date/Time."
                    )

            st.subheader("Selected Machine Summary")

            c1, c2, c3, c4, c5, c6 = st.columns(6)

            # Requirement: Screw RPM summary should consider only RPM values greater than 0.
            positive_rpm_data = selected_data[pd.to_numeric(selected_data["rpm"], errors="coerce") > 0].copy() if not selected_data.empty else pd.DataFrame()

            c1.metric("Total Records", len(selected_data))
            c2.metric("Min Diameter", int(selected_data["diameter"].min()) if not selected_data.empty else 0)
            c3.metric("Max Diameter", int(selected_data["diameter"].max()) if not selected_data.empty else 0)
            c4.metric("Min Screw RPM", int(positive_rpm_data["rpm"].min()) if not positive_rpm_data.empty else 0)
            c5.metric("Max Screw RPM", int(positive_rpm_data["rpm"].max()) if not positive_rpm_data.empty else 0)
            c6.metric("Machines in Range", len(available_machines_in_range))

            if selected_data.empty:
                st.warning("No data available for selected machine(s) and selected date/time range.")
                st.stop()

            st.subheader("Continuous Diameter Zone Table")

            st.write("""
            Logic used:
            - Diameter decimal readings are converted to integer zone values.
            - Same continuous diameter value is treated as one zone.
            - When diameter changes, a new zone starts.
            - Only zones with **more than 30 minutes** are considered.
            - For each zone, Diameter, RPM and Speed values greater than 0 only are considered.
            - Inside each same-diameter zone, the data is split by Speed + Screw RPM.
            - Every continuous Diameter + Speed + RPM segment greater than 30 minutes is shown.
            - Start time, end time and duration are calculated only for each exact Speed/RPM continuous segment, not the full diameter zone.
            - Date/time filtering uses the actual timestamp column from the uploaded CSV/ZIP data.
            """)

            if zone_result.empty:
                st.warning("No valid zones found. Reason: no continuous same-diameter segment has more than 30 minutes in the selected date/time range.")
            else:
                visible_cols = [
                    "Zone No",
                    "Diameter",
                    "Screw RPM Min",
                    "Screw RPM Max",
                    "Most Repeated Speed",
                    "RPM at Most Repeated Speed",
                    "Speed at RPM Max",
                    "Start Timestamp",
                    "End Timestamp",
                    "Duration",
                    "Duration Minutes",
                    "Readings Count",
                    "Data Pair",
                    "RPM Min Timestamp",
                    "RPM Max Timestamp",
                    "Diameter Source",
                    "RPM Source"
                ]

                if len(selected_machines) > 1:
                    visible_cols = ["Machine"] + visible_cols

                st.dataframe(zone_result[visible_cols], width="stretch")

                st.subheader("RPM Max Ascending Zone Table")
                st.caption(
                    "This table is sorted by Machine and Screw RPM Max in ascending order. "
                    "Only Diameter, Screw RPM and Speed values greater than 0 are considered. "
                    "All valid Speed + RPM segments are shown using the same logic for every selected machine."
                )

                if rpm_ascending_result.empty:
                    st.warning("No RPM Max Ascending rows found because Screw RPM Max or Speed values are 0/blank.")
                else:
                    st.dataframe(rpm_ascending_result, width="stretch")

                st.subheader("Same Diameter Comparison Table")
                st.caption(
                    "This table compares all machines selected in the Zone Analysis Dropdown. "
                    "Only same diameter values available in all selected machines are shown. "
                    "All valid Speed + RPM segments greater than 0 are shown for each common diameter."
                )

                if len(selected_machines) < 2:
                    st.info("Select two or more machines to compare same diameter values across machines.")
                elif same_diameter_result.empty:
                    st.warning("No same diameter values found across all selected machines in the selected date/time range.")
                else:
                    st.dataframe(same_diameter_result, width="stretch")

                st.subheader("Graph 1: Zone Duration by Diameter")

                fig_duration = px.bar(
                    zone_result,
                    x="Zone No",
                    y="Duration Minutes",
                    color="Machine" if len(selected_machines) > 1 else "Diameter",
                    hover_data=[
                        "Machine",
                        "Diameter",
                        "Screw RPM Min",
                        "Screw RPM Max",
                        "Speed at RPM Max",
                        "Readings Count",
                        "Start Timestamp",
                        "End Timestamp",
                        "Data Pair"
                    ],
                    title="Continuous Zone Duration"
                )
                st.plotly_chart(fig_duration, width="stretch")

                st.subheader("Graph 2: Screw RPM Max Ascending by Zone")

                if rpm_ascending_result.empty:
                    st.info("Screw RPM Max graph is not shown because there are no RPM values greater than 0 in the RPM Max Ascending Zone Table.")
                else:
                    fig_rpm_max = px.bar(
                        rpm_ascending_result,
                        x="Zone No",
                        y="Screw RPM Max",
                        color="Machine" if len(selected_machines) > 1 else "Diameter",
                        hover_data=[
                            "Machine",
                            "Diameter",
                            "Speed",
                            "Start Timestamp",
                            "End Timestamp",
                            "Duration",
                            "Data Pair"
                        ],
                        title="Screw RPM Max - Ascending Zone View"
                    )
                    st.plotly_chart(fig_rpm_max, width="stretch")

            st.subheader("Graph 3: Diameter Trend for Selected Machine(s)")

            fig_dia = px.line(
                selected_data,
                x="timestamp",
                y="diameter",
                color="machine_name" if len(selected_machines) > 1 else "data_pair",
                title="Integer Diameter vs Time"
            )
            st.plotly_chart(fig_dia, width="stretch")

            st.subheader("Graph 4: Screw RPM Trend for Selected Machine(s)")

            fig_rpm = px.line(
                selected_data,
                x="timestamp",
                y="rpm",
                color="machine_name" if len(selected_machines) > 1 else "data_pair",
                title="Integer Screw RPM vs Time"
            )
            st.plotly_chart(fig_rpm, width="stretch")

            st.subheader("Selected Machine Raw Data")
            st.dataframe(selected_data, width="stretch")

            excel_data = convert_to_excel(
                zone_result,
                rpm_ascending_result,
                same_diameter_result,
                selected_data,
                match_table
            )

            selected_machine_file_name = "_".join([make_machine_key(m) for m in selected_machines])

            st.download_button(
                label="Download Selected Machine Zone Analysis Excel",
                data=excel_data,
                file_name=f"{selected_machine_file_name}_continuous_diameter_zone_analysis.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )

    # =========================================================
    # TAB 4 - DATABASE CONTROL
    # =========================================================

    with tab4:
        st.header("Database Control")

        db_data = load_database()
        saved_machines = load_saved_machine_list()
        show_database_error_if_any()

        st.write("Database storage mode:")

        if USE_EXTERNAL_DATABASE:
            st.success("Persistent cloud database is connected through DATABASE_URL.")
            st.code("External PostgreSQL / cloud database")
        else:
            st.warning("Local SQLite fallback is active. On Streamlit Cloud, this is not permanent.")
            st.code(str(DB_PATH))

        st.metric("Total Stored Rows", len(db_data))
        st.metric("Frozen Machines Count", len(saved_machines))

        st.info("""
        Important: If you already imported data with an older code version, use **Clear Machine Data Database**
        and import the CSV/ZIP files again. This will apply the corrected RPM logic.

        Correct RPM logic: only **screw_rpm_1, screw_rpm_2, screw_rpm_3** are used.
        **bi_color_rpm is excluded** from Screw RPM analysis.
        """)

        if saved_machines:
            st.subheader("Frozen Machine List")
            st.dataframe(
                pd.DataFrame({
                    "Machine Name": saved_machines,
                    "Machine Key": [make_machine_key(m) for m in saved_machines]
                }),
                width="stretch"
            )

        if not db_data.empty:
            st.subheader("Imported Machine Names in Database")

            imported_names = sorted(db_data["machine_name"].dropna().unique())

            st.dataframe(
                pd.DataFrame({
                    "Imported Machine Name": imported_names,
                    "Machine Key": [make_machine_key(m) for m in imported_names]
                }),
                width="stretch"
            )

            st.subheader("Source Columns Used")

            available_cols = [
                "machine_name",
                "machine_key",
                "diameter_source_column",
                "rpm_source_column",
                "data_pair",
                "speed_source_column",
                "source_file"
            ]

            existing_cols = [col for col in available_cols if col in db_data.columns]
            source_cols = db_data[existing_cols].drop_duplicates()

            st.dataframe(source_cols, width="stretch")

            st.subheader("Stored Data Preview")
            st.dataframe(db_data.tail(100), width="stretch")

        st.warning("Use Clear Database only if you want to delete all imported machine data.")

        confirm_clear = st.checkbox("I confirm I want to clear all imported machine data")

        if confirm_clear:
            if st.button("Clear Machine Data Database"):
                clear_database()
                st.success("Machine data database cleared successfully. Please refresh the app.")


if __name__ == '__main__':
    try:
        main()
    except Exception as app_error:
        st.error('The app found an error, but it did not crash. Please share the technical details below.')
        with st.expander('Show technical error details'):
            st.code(traceback.format_exc())
        st.info('Usually this is caused by DATABASE_URL, Supabase password/host, missing requirements, or a data format issue. After fixing, reboot the app.')
