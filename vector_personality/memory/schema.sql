-- Vector Personality Project - SQL Server Schema
-- Phase 1: Memory Foundation
-- Created: 2025-12-05
-- Version: 1.0.0

-- Principle II: Persistent Memory Architecture
-- This schema supports dual-tier memory (working + long-term)

USE vector_memory;
GO

-- Drop tables in correct order (child tables first due to foreign keys)
IF OBJECT_ID('conversations', 'U') IS NOT NULL DROP TABLE conversations;
IF OBJECT_ID('objects', 'U') IS NOT NULL DROP TABLE objects;
IF OBJECT_ID('rooms', 'U') IS NOT NULL DROP TABLE rooms;
IF OBJECT_ID('faces', 'U') IS NOT NULL DROP TABLE faces;
IF OBJECT_ID('personality_learned', 'U') IS NOT NULL DROP TABLE personality_learned;
IF OBJECT_ID('schema_version', 'U') IS NOT NULL DROP TABLE schema_version;
GO

-- ============================================================
-- TABLE: faces
-- Purpose: Store all face encounters with interaction history
-- Principle: Authentic Perception (I) + Persistent Memory (II)
-- ============================================================

CREATE TABLE faces (
    face_id UNIQUEIDENTIFIER PRIMARY KEY DEFAULT NEWID(),
    name NVARCHAR(100) NULL,
    first_seen DATETIME NOT NULL DEFAULT GETDATE(),
    last_seen DATETIME NOT NULL DEFAULT GETDATE(),
    total_interactions INT DEFAULT 0 CHECK (total_interactions >= 0),
    last_mood_change INT DEFAULT 0 CHECK (last_mood_change BETWEEN -100 AND 100),
    notes NVARCHAR(MAX) NULL,  -- JSON for learned facts about this person
    merged_to UNIQUEIDENTIFIER NULL, -- Points to canonical face_id if merged
    merged_at DATETIME NULL,
    created_at DATETIME NOT NULL DEFAULT GETDATE(),
    updated_at DATETIME NOT NULL DEFAULT GETDATE()
);

CREATE INDEX idx_faces_name ON faces(name);
CREATE INDEX idx_faces_last_seen ON faces(last_seen DESC);

-- ============================================================
-- TABLE: face_embeddings
-- Purpose: Store face embeddings for matching and deduplication
-- ============================================================

CREATE TABLE face_embeddings (
    embedding_id UNIQUEIDENTIFIER PRIMARY KEY DEFAULT NEWID(),
    face_id UNIQUEIDENTIFIER NOT NULL,
    embedding VARBINARY(MAX) NOT NULL,
    vector_dim INT NOT NULL,
    created_at DATETIME NOT NULL DEFAULT GETDATE(),
    FOREIGN KEY (face_id) REFERENCES faces(face_id) ON DELETE CASCADE
);

CREATE INDEX idx_face_embeddings_face_id ON face_embeddings(face_id);

GO

-- ============================================================
-- TABLE: rooms
-- Purpose: Store room profiles and context-aware behavior adjustments
-- Principle: Authentic Perception (I) + Programmable Personality (V)
-- ============================================================

CREATE TABLE rooms (
    room_id UNIQUEIDENTIFIER PRIMARY KEY DEFAULT NEWID(),
    room_name NVARCHAR(100) UNIQUE NULL,
    typical_objects NVARCHAR(MAX) NULL,  -- JSON list of common objects
    context_behavior_adjustments NVARCHAR(MAX) NULL,  -- JSON: {"quietness": 0.8}
    first_identified DATETIME NOT NULL DEFAULT GETDATE(),
    last_visited DATETIME NOT NULL DEFAULT GETDATE(),
    visit_count INT DEFAULT 0 CHECK (visit_count >= 0),
    created_at DATETIME NOT NULL DEFAULT GETDATE(),
    updated_at DATETIME NOT NULL DEFAULT GETDATE(),
    CONSTRAINT ck_room_identification CHECK (room_name IS NOT NULL OR typical_objects IS NOT NULL)
);

CREATE INDEX idx_rooms_name ON rooms(room_name);
CREATE INDEX idx_rooms_last_visited ON rooms(last_visited DESC);

GO

-- ============================================================
-- TABLE: objects
-- Purpose: Store detected objects with location and room context
-- Principle: Authentic Perception (I)
-- ============================================================

CREATE TABLE objects (
    object_id UNIQUEIDENTIFIER PRIMARY KEY DEFAULT NEWID(),
    object_type NVARCHAR(100) NOT NULL,
    room_id UNIQUEIDENTIFIER NULL,
    confidence FLOAT NOT NULL CHECK (confidence BETWEEN 0.0 AND 1.0),
    location_description NVARCHAR(500) NULL,  -- "on the desk", "near window"
    first_detected DATETIME NOT NULL DEFAULT GETDATE(),
    last_detected DATETIME NOT NULL DEFAULT GETDATE(),
    detection_count INT DEFAULT 1 CHECK (detection_count > 0),
    created_at DATETIME NOT NULL DEFAULT GETDATE(),
    updated_at DATETIME NOT NULL DEFAULT GETDATE(),
    FOREIGN KEY (room_id) REFERENCES rooms(room_id) ON DELETE SET NULL
);

CREATE INDEX idx_objects_type ON objects(object_type);
CREATE INDEX idx_objects_room ON objects(room_id);
CREATE INDEX idx_objects_last_detected ON objects(last_detected DESC);

GO

-- ============================================================
-- TABLE: conversations
-- Purpose: Store all conversation turns with emotional context
-- Principle: Persistent Memory (II) + Emotional Authenticity (III)
-- ============================================================

CREATE TABLE conversations (
    conversation_id UNIQUEIDENTIFIER PRIMARY KEY DEFAULT NEWID(),
    speaker_id UNIQUEIDENTIFIER NOT NULL,
    timestamp DATETIME NOT NULL DEFAULT GETDATE(),
    text NVARCHAR(MAX) NOT NULL,
    room_id UNIQUEIDENTIFIER NULL,
    emotional_context INT DEFAULT 50 CHECK (emotional_context BETWEEN 0 AND 100),
    response_text NVARCHAR(MAX) NULL,  -- Vector's response
    response_type NVARCHAR(50) NULL,  -- 'sdk', 'api_cheap', 'api_moderate', 'api_expensive'
    created_at DATETIME NOT NULL DEFAULT GETDATE(),
    FOREIGN KEY (speaker_id) REFERENCES faces(face_id) ON DELETE CASCADE,
    FOREIGN KEY (room_id) REFERENCES rooms(room_id) ON DELETE SET NULL
);

CREATE INDEX idx_conversations_speaker ON conversations(speaker_id);
CREATE INDEX idx_conversations_timestamp ON conversations(timestamp DESC);
CREATE INDEX idx_conversations_room ON conversations(room_id);

-- Combined index to speed up date-range + speaker retrieval, including the searched text columns
CREATE INDEX idx_conversations_search ON conversations(timestamp DESC, speaker_id) INCLUDE (text, response_text);

-- NOTE: Full-text search is optional; it requires SQL Server Full-Text feature enabled.
-- CREATE FULLTEXT INDEX ON conversations(text, response_text) KEY INDEX PK__conversa__...;

GO

-- ============================================================
-- TABLE: personality_learned
-- Purpose: Store learned personality trait adjustments
-- Principle: Programmable Personality (V)
-- ============================================================

CREATE TABLE personality_learned (
    id INT IDENTITY(1,1) PRIMARY KEY,
    timestamp DATETIME NOT NULL DEFAULT GETDATE(),
    curiosity_delta FLOAT DEFAULT 0.0 CHECK (curiosity_delta BETWEEN -1.0 AND 1.0),
    touchiness_delta FLOAT DEFAULT 0.0 CHECK (touchiness_delta BETWEEN -1.0 AND 1.0),
    vitality_delta FLOAT DEFAULT 0.0 CHECK (vitality_delta BETWEEN -1.0 AND 1.0),
    friendliness_delta FLOAT DEFAULT 0.0 CHECK (friendliness_delta BETWEEN -1.0 AND 1.0),
    courage_delta FLOAT DEFAULT 0.0 CHECK (courage_delta BETWEEN -1.0 AND 1.0),
    sassiness_delta FLOAT DEFAULT 0.0 CHECK (sassiness_delta BETWEEN -1.0 AND 1.0),
    feedback_text NVARCHAR(500) NULL,  -- User feedback that triggered adjustment
    created_at DATETIME NOT NULL DEFAULT GETDATE()
);

CREATE INDEX idx_personality_timestamp ON personality_learned(timestamp DESC);

GO

-- ============================================================
-- STORED PROCEDURES & FUNCTIONS
-- ============================================================

-- Get current cumulative personality deltas
CREATE OR ALTER PROCEDURE GetCurrentPersonalityDeltas
AS
BEGIN
    SELECT 
        SUM(curiosity_delta) AS curiosity_delta,
        SUM(touchiness_delta) AS touchiness_delta,
        SUM(vitality_delta) AS vitality_delta,
        SUM(friendliness_delta) AS friendliness_delta,
        SUM(courage_delta) AS courage_delta,
        SUM(sassiness_delta) AS sassiness_delta
    FROM personality_learned
END;
GO

-- Update face last_seen timestamp and increment interactions
CREATE OR ALTER PROCEDURE UpdateFaceInteraction
    @face_id UNIQUEIDENTIFIER,
    @mood_change INT = 0
AS
BEGIN
    UPDATE faces
    SET 
        last_seen = GETDATE(),
        total_interactions = total_interactions + 1,
        last_mood_change = @mood_change,
        updated_at = GETDATE()
    WHERE face_id = @face_id
END;
GO

-- Get room by name or create if not exists
CREATE OR ALTER PROCEDURE GetOrCreateRoom
    @room_name NVARCHAR(100)
AS
BEGIN
    DECLARE @room_id UNIQUEIDENTIFIER
    
    SELECT @room_id = room_id FROM rooms WHERE room_name = @room_name
    
    IF @room_id IS NULL
    BEGIN
        DECLARE @new_room_id UNIQUEIDENTIFIER = NEWID()
        INSERT INTO rooms (room_id, room_name)
        VALUES (@new_room_id, @room_name)
        SET @room_id = @new_room_id
    END
    ELSE
    BEGIN
        UPDATE rooms
        SET 
            last_visited = GETDATE(),
            visit_count = visit_count + 1,
            updated_at = GETDATE()
        WHERE room_id = @room_id
    END
    
    SELECT * FROM rooms WHERE room_id = @room_id
END;
GO

-- ============================================================
-- INITIAL DATA SEEDING
-- ============================================================

-- ============================================================
-- SCHEMA VERSION TRACKING
-- ============================================================

CREATE TABLE schema_version (
    version NVARCHAR(20) PRIMARY KEY,
    applied_at DATETIME NOT NULL DEFAULT GETDATE(),
    description NVARCHAR(500) NULL
);

INSERT INTO schema_version (version, description)
VALUES ('1.0.0', 'Initial schema: faces, objects, rooms, conversations, budget_usage, personality_learned');

GO

PRINT 'Vector Personality Schema v1.0.0 applied successfully!';
PRINT 'Tables created: faces, objects, rooms, conversations, budget_usage, personality_learned, schema_version';
PRINT 'Stored procedures created: GetCurrentPersonalityDeltas, GetOrCreateBudgetEntry, UpdateFaceInteraction, GetOrCreateRoom';
GO
