"""Database connection for the (hidden) PSR menu item.

Java uses JDBC with com.mysql.jdbc.Driver. Here any DB-API 2 module works:
pymysql or mysql.connector for MySQL (if installed), sqlite3 otherwise.
The driver argument picks the module, e.g. "pymysql", "mysql.connector",
"sqlite3"; the java driver name "com.mysql.jdbc.Driver" maps to whichever
MySQL module is found.
"""

import importlib

from . import ui_hooks

DEFAULT_DB = 1
MYSQL = 1

_MYSQL_MODULES = ("pymysql", "mysql.connector", "MySQLdb")


def _load_driver(driver):
    if driver in ("com.mysql.jdbc.Driver", "com.mysql.cj.jdbc.Driver", "mysql"):
        for m in _MYSQL_MODULES:
            try:
                return importlib.import_module(m), "mysql"
            except ImportError:
                pass
        raise ImportError("no MySQL driver found, install pymysql (pip install pymysql)")
    mod = importlib.import_module(driver)
    return mod, ("sqlite" if driver == "sqlite3" else "mysql" if driver in _MYSQL_MODULES else driver)


class DatabaseConnection:
    def __init__(self, driver, url, host, database, user, password):
        self._connection = None
        self._driver = driver
        self._url = url
        self._host = host or "localhost"
        self._database = database
        self._user = user
        self._password = password
        self._product = ""
        print("Connecting using user '" + self._user + "'")

    def open_connection(self):
        mod, kind = _load_driver(self._driver)
        if kind == "sqlite":
            self._connection = mod.connect(self._url or self._database)
            self._product = "SQLite"
        else:
            host, port = self._host, 3306
            # jdbc:mysql://host:port/db
            if self._url.startswith("jdbc:mysql://"):
                rest = self._url[len("jdbc:mysql://"):]
                hp, _, db = rest.partition("/")
                host, _, p = hp.partition(":")
                if p:
                    port = int(p)
                if db:
                    self._database = db
            kw = dict(host=host, port=port, user=self._user, password=self._password, database=self._database)
            self._connection = mod.connect(**kw)
            self._product = "MySQL"
        print(self._product + " " + self._database + "  connection open")

    def close_connection(self):
        try:
            print(self._product + " " + self._database + "  connection lost")
            if self._connection is not None:
                self._connection.close()
            self._connection = None
        except Exception as e:
            ui_hooks.show_message("alert", "Could not close connection!\n" + str(e), kind="error")

    def execute_statement(self, query, is_simple=True, print_sql=False):
        """returns an executed cursor, or None"""
        try:
            cur = self._connection.cursor()
        except Exception:
            print("Connection timed out during createStatement. Re-establishing.")
            try:
                self.open_connection()
                cur = self._connection.cursor()
            except Exception:
                print("Reconnect failed.")
                return None
        try:
            if print_sql:
                print(query)
            cur.execute(query)
            return cur
        except Exception as e3:
            if "onnection" in str(e3):
                print("Connection timed out. Re-establishing.")
            try:
                self.open_connection()
                cur = self._connection.cursor()
                cur.execute(query)
                return cur
            except Exception as e4:
                if "onnection" in str(e3):
                    print("Reconnect failed.")
                ui_hooks.show_message("alert", "Could not execute statement!\n" + str(e4), kind="error")
                return None

    def execute_update(self, query):
        try:
            cur = self._connection.cursor()
            print(query)
            cur.execute(query)
            self._connection.commit()
            return True
        except Exception as e:
            ui_hooks.show_message("alert", "Could not execute statement!\n" + str(e), kind="error")
            return False

    def execute(self, query, is_print_query=False):
        try:
            cur = self._connection.cursor()
            if is_print_query:
                print(query)
            cur.execute(query)
            return True
        except Exception as e:
            ui_hooks.show_message("alert", "Could not execute statement!\n" + str(e), kind="error")
            return False

    def get_create_table_query(self, table_name, select_clause, remainder_clause):
        if self.get_database_product() == MYSQL:
            return "CREATE TABLE " + table_name + " AS SELECT " + select_clause + " " + remainder_clause + ";"
        return "SELECT " + select_clause + " INTO " + table_name + " " + remainder_clause + ";"

    def get_database_product(self):
        print("DB = '" + self._product + "'")
        return MYSQL if self._product.lower() == "mysql" else DEFAULT_DB

    def supports_create_index(self):
        return True

    def get_table_view_names(self):
        if self._product == "SQLite":
            q = "SELECT name FROM sqlite_master WHERE type IN ('table', 'view')"
        else:
            q = "SHOW FULL TABLES"
        cur = self._connection.cursor()
        cur.execute(q)
        return [r[0] for r in cur.fetchall()]

    def check_table_name(self, table_name):
        try:
            existing = self.get_table_view_names()
            if table_name != "":
                return table_name not in existing
            return True
        except Exception as e:
            ui_hooks.show_message("alert", "Table name already exists!\n" + str(e), kind="error")
            return False

    def set_driver(self, v):
        self._driver = v

    def set_url(self, v):
        self._url = v

    def set_host(self, v):
        self._host = v

    def set_database(self, v):
        self._database = v

    def set_user(self, v):
        self._user = v

    def set_password(self, v):
        self._password = v
