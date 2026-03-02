-- OP-ECOM Analytics Tracker Schema
-- MariaDB/MySQL Database Schema
-- Database: op_ecom_tracker

-- Create database if not exists
CREATE DATABASE IF NOT EXISTS op_ecom_tracker;
USE op_ecom_tracker;

-- ============================================
-- Table: sessions
-- One row per visitor session
-- ============================================
CREATE TABLE IF NOT EXISTS sessions (
    id INT AUTO_INCREMENT PRIMARY KEY,
    session_id VARCHAR(64) UNIQUE NOT NULL,
    visitor_type ENUM('New_Visitor', 'Returning_Visitor', 'Other') DEFAULT 'New_Visitor',
    browser VARCHAR(50),
    operating_system VARCHAR(50),
    region INT DEFAULT 1,
    traffic_type INT DEFAULT 1,
    is_weekend BOOLEAN DEFAULT FALSE,
    month VARCHAR(10),
    special_day FLOAT DEFAULT 0.0,
    started_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    ended_at TIMESTAMP NULL,
    revenue BOOLEAN DEFAULT FALSE,
    
    -- Aggregated metrics (updated on session end)
    administrative_count INT DEFAULT 0,
    administrative_duration FLOAT DEFAULT 0.0,
    informational_count INT DEFAULT 0,
    informational_duration FLOAT DEFAULT 0.0,
    product_related_count INT DEFAULT 0,
    product_related_duration FLOAT DEFAULT 0.0,
    bounce_rates FLOAT DEFAULT 0.0,
    exit_rates FLOAT DEFAULT 0.0,
    page_values FLOAT DEFAULT 0.0,
    
    INDEX idx_session_id (session_id),
    INDEX idx_started_at (started_at)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

-- ============================================
-- Table: page_views
-- Each page visit within a session
-- ============================================
CREATE TABLE IF NOT EXISTS page_views (
    id INT AUTO_INCREMENT PRIMARY KEY,
    session_id VARCHAR(64) NOT NULL,
    page_type ENUM('Administrative', 'Informational', 'ProductRelated') DEFAULT 'ProductRelated',
    page_url VARCHAR(500),
    page_title VARCHAR(255),
    duration_seconds FLOAT DEFAULT 0.0,
    is_bounce BOOLEAN DEFAULT FALSE,
    is_exit BOOLEAN DEFAULT FALSE,
    page_value FLOAT DEFAULT 0.0,
    scroll_depth INT DEFAULT 0,
    viewed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    
    INDEX idx_page_session (session_id),
    FOREIGN KEY (session_id) REFERENCES sessions(session_id) ON DELETE CASCADE
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

-- ============================================
-- Table: events
-- Custom events (clicks, purchases, etc.)
-- ============================================
CREATE TABLE IF NOT EXISTS events (
    id INT AUTO_INCREMENT PRIMARY KEY,
    session_id VARCHAR(64) NOT NULL,
    event_type VARCHAR(50) NOT NULL,
    event_category VARCHAR(50),
    event_label VARCHAR(255),
    event_value FLOAT DEFAULT 0.0,
    event_data JSON,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    
    INDEX idx_event_session (session_id),
    INDEX idx_event_type (event_type),
    FOREIGN KEY (session_id) REFERENCES sessions(session_id) ON DELETE CASCADE
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

-- ============================================
-- View: sessions_export
-- Export view matching UCI dataset format
-- ============================================
CREATE OR REPLACE VIEW sessions_export AS
SELECT 
    administrative_count AS Administrative,
    administrative_duration AS Administrative_Duration,
    informational_count AS Informational,
    informational_duration AS Informational_Duration,
    product_related_count AS ProductRelated,
    product_related_duration AS ProductRelated_Duration,
    bounce_rates AS BounceRates,
    exit_rates AS ExitRates,
    page_values AS PageValues,
    special_day AS SpecialDay,
    month AS Month,
    operating_system AS OperatingSystems,
    browser AS Browser,
    region AS Region,
    traffic_type AS TrafficType,
    visitor_type AS VisitorType,
    is_weekend AS Weekend,
    revenue AS Revenue
FROM sessions
WHERE ended_at IS NOT NULL;
