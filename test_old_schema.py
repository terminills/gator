#!/usr/bin/env python3
"""
Create a test database with old schema (without ai_domain and ai_subdomain columns)
to test the migration.
"""

import sqlite3
import sys
import os

def create_old_schema_database():
    """Create a database with the old acd_contexts schema."""
    
    db_path = "gator_old_schema.db"
    
    # Remove existing test database
    if os.path.exists(db_path):
        os.remove(db_path)
    
    print("Creating test database with old schema...")
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    # Create acd_contexts table WITHOUT ai_domain and ai_subdomain
    cursor.execute("""
        CREATE TABLE acd_contexts (
            id UUID PRIMARY KEY,
            benchmark_id UUID,
            content_id UUID,
            ai_phase VARCHAR(100) NOT NULL,
            ai_status VARCHAR(20) NOT NULL,
            ai_complexity VARCHAR(20),
            ai_note TEXT,
            ai_dependencies JSON,
            ai_commit VARCHAR(40),
            ai_commit_history JSON,
            ai_version VARCHAR(50),
            ai_change TEXT,
            ai_pattern VARCHAR(100),
            ai_strategy TEXT,
            ai_train_hash VARCHAR(64),
            ai_context JSON,
            ai_metadata JSON,
            compiler_err TEXT,
            runtime_err TEXT,
            fix_reason TEXT,
            human_override TEXT,
            ai_assigned_to VARCHAR(100),
            ai_assigned_by VARCHAR(100),
            ai_assigned_at DATETIME,
            ai_assignment_reason TEXT,
            ai_previous_assignee VARCHAR(100),
            ai_assignment_history JSON,
            ai_handoff_requested BOOLEAN DEFAULT 0,
            ai_handoff_reason TEXT,
            ai_handoff_to VARCHAR(100),
            ai_handoff_type VARCHAR(20),
            ai_handoff_at DATETIME,
            ai_handoff_notes TEXT,
            ai_handoff_status VARCHAR(20),
            ai_required_capabilities JSON,
            ai_preferred_agent_type VARCHAR(100),
            ai_agent_pool JSON,
            ai_skill_level_required VARCHAR(20),
            ai_timeout INTEGER,
            ai_max_retries INTEGER,
            ai_confidence VARCHAR(20),
            ai_request VARCHAR(30),
            ai_state VARCHAR(20) NOT NULL DEFAULT 'READY',
            ai_note_confidence TEXT,
            ai_request_from VARCHAR(100),
            ai_note_request TEXT,
            ai_queue_priority VARCHAR(20) DEFAULT 'NORMAL',
            ai_queue_status VARCHAR(30) DEFAULT 'QUEUED',
            ai_queue_reason TEXT,
            ai_started DATETIME,
            ai_estimated_completion DATETIME,
            ai_validation VARCHAR(20),
            ai_issues JSON,
            ai_suggestions JSON,
            ai_refinement VARCHAR(20),
            ai_changes TEXT,
            ai_rationale TEXT,
            ai_validation_result VARCHAR(20),
            ai_approval VARCHAR(20),
            ai_exchange_id VARCHAR(50),
            ai_round INTEGER,
            created_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
            updated_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP
        )
    """)
    
    # Create some indexes (but not for ai_domain/ai_subdomain since they don't exist)
    cursor.execute("CREATE INDEX ix_acd_contexts_ai_phase ON acd_contexts (ai_phase)")
    cursor.execute("CREATE INDEX ix_acd_contexts_ai_status ON acd_contexts (ai_status)")
    cursor.execute("CREATE INDEX ix_acd_contexts_ai_state ON acd_contexts (ai_state)")
    
    conn.commit()
    conn.close()
    
    print(f"✅ Created old schema database: {db_path}")
    
    # Verify columns
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute("PRAGMA table_info(acd_contexts)")
    columns = [row[1] for row in cursor.fetchall()]
    conn.close()
    
    print(f"\nColumns in database: {len(columns)}")
    print(f"Has ai_domain: {'ai_domain' in columns}")
    print(f"Has ai_subdomain: {'ai_subdomain' in columns}")
    
    if 'ai_domain' not in columns and 'ai_subdomain' not in columns:
        print("\n✅ Old schema database created successfully")
        return db_path
    else:
        print("\n❌ Failed to create old schema database")
        return None


if __name__ == "__main__":
    create_old_schema_database()
