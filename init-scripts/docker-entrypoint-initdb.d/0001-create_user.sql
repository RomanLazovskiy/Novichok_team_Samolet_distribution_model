-- create_user.sql
CREATE USER novichok WITH PASSWORD 'qwerty123';
ALTER ROLE novichok SET client_encoding TO 'utf8';
ALTER ROLE novichok SET default_transaction_isolation TO 'read committed';
ALTER ROLE novichok SET timezone TO 'UTC';