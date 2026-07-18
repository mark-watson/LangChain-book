"""Build a small self-contained SQLite database for the SQL agent chapter.

Four tables (departments, employees, customers, invoices) with a couple
dozen rows total — small enough to eyeball, big enough that the queries
the agent has to write are non-trivial.
"""

import sqlite3
from pathlib import Path

DB_PATH = Path(__file__).parent / "company.db"


def make_db() -> Path:
    DB_PATH.unlink(missing_ok=True)
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()

    c.executescript(
        """
        CREATE TABLE departments (
            id   INTEGER PRIMARY KEY,
            name TEXT NOT NULL
        );

        CREATE TABLE employees (
            id            INTEGER PRIMARY KEY,
            first_name    TEXT NOT NULL,
            last_name     TEXT NOT NULL,
            department_id INTEGER NOT NULL,
            hire_date     TEXT NOT NULL,
            salary        REAL NOT NULL,
            FOREIGN KEY (department_id) REFERENCES departments(id)
        );

        CREATE TABLE customers (
            id         INTEGER PRIMARY KEY,
            first_name TEXT NOT NULL,
            last_name  TEXT NOT NULL,
            city       TEXT,
            country    TEXT
        );

        CREATE TABLE invoices (
            id           INTEGER PRIMARY KEY,
            customer_id  INTEGER NOT NULL,
            employee_id  INTEGER NOT NULL,
            invoice_date TEXT NOT NULL,
            total        REAL NOT NULL,
            FOREIGN KEY (customer_id) REFERENCES customers(id),
            FOREIGN KEY (employee_id) REFERENCES employees(id)
        );
        """
    )

    c.executemany(
        "INSERT INTO departments VALUES (?, ?)",
        [(1, "Engineering"), (2, "Sales"), (3, "Support")],
    )

    c.executemany(
        "INSERT INTO employees VALUES (?, ?, ?, ?, ?, ?)",
        [
            (1, "Alice", "Anderson", 1, "2020-01-15", 95000),
            (2, "Bob", "Brown", 1, "2019-03-20", 110000),
            (3, "Carol", "Chen", 2, "2021-06-10", 82000),
            (4, "Dan", "Davis", 2, "2018-11-05", 88000),
            (5, "Eve", "Evans", 3, "2022-02-28", 65000),
        ],
    )

    c.executemany(
        "INSERT INTO customers VALUES (?, ?, ?, ?, ?)",
        [
            (1, "John", "Smith", "New York", "USA"),
            (2, "Marie", "Dupont", "Paris", "France"),
            (3, "Hans", "Mueller", "Berlin", "Germany"),
            (4, "Yuki", "Tanaka", "Tokyo", "Japan"),
            (5, "Sofia", "Rossi", "Rome", "Italy"),
            (6, "Emma", "Johnson", "Chicago", "USA"),
        ],
    )

    c.executemany(
        "INSERT INTO invoices VALUES (?, ?, ?, ?, ?)",
        [
            (1, 1, 3, "2024-01-15", 1250.00),
            (2, 1, 3, "2024-03-20", 850.00),
            (3, 2, 4, "2024-02-10", 2100.00),
            (4, 3, 3, "2024-04-05", 675.00),
            (5, 4, 4, "2024-05-12", 3200.00),
            (6, 5, 3, "2024-06-01", 450.00),
            (7, 6, 4, "2024-07-15", 1800.00),
            (8, 1, 3, "2024-08-22", 950.00),
            (9, 2, 4, "2024-09-10", 1600.00),
        ],
    )

    conn.commit()
    conn.close()
    return DB_PATH


if __name__ == "__main__":
    path = make_db()
    print(f"Created {path}")
