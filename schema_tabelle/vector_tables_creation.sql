
CREATE TABLE RobotStates (
    Id INT IDENTITY(1,1) PRIMARY KEY,
    Timestamp DATETIME DEFAULT GETDATE(),
    EventType NVARCHAR(100),
    Description NVARCHAR(MAX)
);

CREATE TABLE RobotFaces (
    Id INT IDENTITY(1,1) PRIMARY KEY,
    Timestamp DATETIME DEFAULT GETDATE(),
    EventType NVARCHAR(100),
    Description NVARCHAR(MAX)
);

CREATE TABLE WakeWordEvents (
    Id INT IDENTITY(1,1) PRIMARY KEY,
    Timestamp DATETIME DEFAULT GETDATE(),
    EventType NVARCHAR(100),
    Description NVARCHAR(MAX)
);

CREATE TABLE UserIntents (
    Id INT IDENTITY(1,1) PRIMARY KEY,
    Timestamp DATETIME DEFAULT GETDATE(),
    EventType NVARCHAR(100),
    Description NVARCHAR(MAX)
);

CREATE TABLE CubeTaps (
    Id INT IDENTITY(1,1) PRIMARY KEY,
    Timestamp DATETIME DEFAULT GETDATE(),
    EventType NVARCHAR(100),
    Description NVARCHAR(MAX)
);

CREATE TABLE ObjectMovementEvents (
    Id INT IDENTITY(1,1) PRIMARY KEY,
    Timestamp DATETIME DEFAULT GETDATE(),
    EventType NVARCHAR(100),
    Description NVARCHAR(MAX)
);

CREATE TABLE ObjectOrientationChanges (
    Id INT IDENTITY(1,1) PRIMARY KEY,
    Timestamp DATETIME DEFAULT GETDATE(),
    EventType NVARCHAR(100),
    Description NVARCHAR(MAX)
);

CREATE TABLE ObservedObjects (
    Id INT IDENTITY(1,1) PRIMARY KEY,
    Timestamp DATETIME DEFAULT GETDATE(),
    EventType NVARCHAR(100),
    Description NVARCHAR(MAX)
);

CREATE TABLE CubeConnections (
    Id INT IDENTITY(1,1) PRIMARY KEY,
    Timestamp DATETIME DEFAULT GETDATE(),
    EventType NVARCHAR(100),
    Description NVARCHAR(MAX)
);

CREATE TABLE CubeConnectionLosses (
    Id INT IDENTITY(1,1) PRIMARY KEY,
    Timestamp DATETIME DEFAULT GETDATE(),
    EventType NVARCHAR(100),
    Description NVARCHAR(MAX)
);

CREATE TABLE MirrorModeEvents (
    Id INT IDENTITY(1,1) PRIMARY KEY,
    Timestamp DATETIME DEFAULT GETDATE(),
    EventType NVARCHAR(100),
    Description NVARCHAR(MAX)
);

CREATE TABLE VisionModeEvents (
    Id INT IDENTITY(1,1) PRIMARY KEY,
    Timestamp DATETIME DEFAULT GETDATE(),
    EventType NVARCHAR(100),
    Description NVARCHAR(MAX)
);

CREATE TABLE CameraImages (
    Id INT IDENTITY(1,1) PRIMARY KEY,
    Timestamp DATETIME DEFAULT GETDATE(),
    EventType NVARCHAR(100),
    Description NVARCHAR(MAX)
);

CREATE TABLE NavMapUpdates (
    Id INT IDENTITY(1,1) PRIMARY KEY,
    Timestamp DATETIME DEFAULT GETDATE(),
    EventType NVARCHAR(100),
    Description NVARCHAR(MAX)
);

CREATE TABLE AudioModeChanges (
    Id INT IDENTITY(1,1) PRIMARY KEY,
    Timestamp DATETIME DEFAULT GETDATE(),
    EventType NVARCHAR(100),
    Description NVARCHAR(MAX)
);
