#!/usr/bin/env bash
# Recreates pharma_sc database + app user from db/schema.sql. Run once per machine.
set -euo pipefail
cd "$(dirname "$0")/.."

if [ -z "${MYSQL_ROOT_PWD:-}" ]; then
  echo "Usage: MYSQL_ROOT_PWD=<root-password> ./db/setup_db.sh" >&2
  exit 1
fi

mysql -h 127.0.0.1 -u root -p"$MYSQL_ROOT_PWD" --default-character-set=utf8mb4 < db/schema.sql
mysql -h 127.0.0.1 -u root -p"$MYSQL_ROOT_PWD" <<'SQL'
CREATE USER IF NOT EXISTS 'pharma_user'@'localhost' IDENTIFIED BY 'pharma_pass';
CREATE USER IF NOT EXISTS 'pharma_user'@'127.0.0.1' IDENTIFIED BY 'pharma_pass';
GRANT ALL PRIVILEGES ON pharma_sc.* TO 'pharma_user'@'localhost';
GRANT ALL PRIVILEGES ON pharma_sc.* TO 'pharma_user'@'127.0.0.1';
FLUSH PRIVILEGES;
SQL

echo "pharma_sc created; app user 'pharma_user' ready."
echo "Seed data:  .venv/bin/python db/db_writer.py --mode seed"
echo "Derived:    .venv/bin/python db/fill_derived.py --full"
