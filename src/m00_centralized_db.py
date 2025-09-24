import sqlite3
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional

class CentralizedDB:
    """Centralized SQLite database for all fine-tuning evaluation modules"""

    def __init__(self, db_path: str = "outputs/centralized.db"):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_database()

    def _init_database(self):
        """Initialize database with complete schema for all modules"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()

            # Core experiment tracking
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS experiments (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    experiment_name TEXT UNIQUE NOT NULL,
                    method TEXT NOT NULL,
                    status TEXT DEFAULT 'pending',
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    config TEXT,
                    notes TEXT
                )
            ''')

            # Module 01: Environment Setup
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS m01_environment (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    python_version TEXT,
                    cuda_version TEXT,
                    torch_version TEXT,
                    transformers_version TEXT,
                    peft_version TEXT,
                    gpu_info TEXT,
                    memory_gb REAL,
                    setup_status TEXT,
                    error_log TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            ''')

            # Module 02: Data Preparation
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS m02_data_preparation (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    dataset_name TEXT NOT NULL,
                    dataset_size INTEGER,
                    train_samples INTEGER,
                    val_samples INTEGER,
                    safe_prompts INTEGER,
                    unsafe_prompts INTEGER,
                    preprocessing_status TEXT,
                    data_hash TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            ''')

            # Module 03a: QLoRA Training
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS m03a_qlora_training (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    experiment_id INTEGER,
                    epoch INTEGER,
                    step INTEGER,
                    training_loss REAL,
                    eval_loss REAL,
                    learning_rate REAL,
                    gradient_norm REAL,
                    memory_usage_mb REAL,
                    lora_r INTEGER,
                    lora_alpha INTEGER,
                    dropout REAL,
                    training_time_minutes REAL,
                    adapter_size_mb REAL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (experiment_id) REFERENCES experiments(id)
                )
            ''')

            # Module 03b: GRIT Training
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS m03b_grit_training (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    experiment_id INTEGER,
                    epoch INTEGER,
                    step INTEGER,
                    training_loss REAL,
                    eval_loss REAL,
                    learning_rate REAL,
                    kfac_damping REAL,
                    reprojection_k INTEGER,
                    eigenvalue_sum REAL,
                    geometry_preservation REAL,
                    memory_usage_mb REAL,
                    training_time_minutes REAL,
                    adapter_size_mb REAL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (experiment_id) REFERENCES experiments(id)
                )
            ''')

            # Module 04: Inference Backends
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS m04_inference_backends (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    experiment_id INTEGER,
                    backend_type TEXT NOT NULL,
                    model_variant TEXT NOT NULL,
                    rps REAL,
                    latency_ms REAL,
                    memory_usage_mb REAL,
                    concurrent_requests INTEGER,
                    deployment_status TEXT,
                    endpoint_url TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (experiment_id) REFERENCES experiments(id)
                )
            ''')

            # Module 05: AQI Evaluation
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS m05_aqi_scores (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    experiment_id INTEGER,
                    model_variant TEXT NOT NULL,
                    xbi_score REAL,
                    chi_score REAL,
                    aqi_score REAL,
                    delta_aqi REAL,
                    safe_prompts_tested INTEGER,
                    unsafe_prompts_tested INTEGER,
                    hidden_states_extracted INTEGER,
                    evaluation_time_minutes REAL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (experiment_id) REFERENCES experiments(id)
                )
            ''')

            cursor.execute('''
                CREATE TABLE IF NOT EXISTS m05_behavioral_metrics (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    experiment_id INTEGER,
                    model_variant TEXT NOT NULL,
                    g_eval_score REAL,
                    safety_refusal_rate REAL,
                    helpfulness_score REAL,
                    alignment_quality REAL,
                    response_quality REAL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (experiment_id) REFERENCES experiments(id)
                )
            ''')

            # Module 06: Comparative Analysis
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS m06_comparative_analysis (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    comparison_type TEXT NOT NULL,
                    method_a TEXT NOT NULL,
                    method_b TEXT NOT NULL,
                    metric_name TEXT NOT NULL,
                    method_a_value REAL,
                    method_b_value REAL,
                    difference REAL,
                    percentage_change REAL,
                    p_value REAL,
                    effect_size REAL,
                    significance TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            ''')

            # Module 07: Results Synthesis
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS m07_results_synthesis (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    report_type TEXT NOT NULL,
                    section TEXT NOT NULL,
                    finding TEXT NOT NULL,
                    recommendation TEXT,
                    confidence_score REAL,
                    supporting_data TEXT,
                    visualization_path TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            ''')

            # Model responses for analysis
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS model_responses (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    experiment_id INTEGER,
                    model_variant TEXT NOT NULL,
                    prompt_id TEXT NOT NULL,
                    prompt_text TEXT NOT NULL,
                    response_text TEXT NOT NULL,
                    prompt_category TEXT,
                    safety_label TEXT,
                    response_time_ms REAL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (experiment_id) REFERENCES experiments(id)
                )
            ''')

            # Hidden states for AQI calculation
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS hidden_states (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    experiment_id INTEGER,
                    model_variant TEXT NOT NULL,
                    prompt_id TEXT NOT NULL,
                    layer_index INTEGER,
                    embeddings BLOB,
                    pooling_method TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (experiment_id) REFERENCES experiments(id)
                )
            ''')

            conn.commit()
            logging.info(f"Database initialized at {self.db_path}")

    def create_experiment(self, name: str, method: str, config: Dict[str, Any] = None) -> int:
        """Create new experiment and return its ID"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute('''
                INSERT OR REPLACE INTO experiments (experiment_name, method, config)
                VALUES (?, ?, ?)
            ''', (name, method, str(config) if config else None))
            return cursor.lastrowid

    def update_experiment_status(self, experiment_id: int, status: str):
        """Update experiment status"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute('''
                UPDATE experiments
                SET status = ?, updated_at = CURRENT_TIMESTAMP
                WHERE id = ?
            ''', (status, experiment_id))

    def log_module_data(self, table: str, data: Dict[str, Any]):
        """Generic method to log data to any module table"""
        if not data:
            return

        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()

            columns = ', '.join(data.keys())
            placeholders = ', '.join(['?' for _ in data])
            values = list(data.values())

            cursor.execute(f'''
                INSERT INTO {table} ({columns})
                VALUES ({placeholders})
            ''', values)

    def get_experiment_data(self, experiment_id: int) -> Dict[str, Any]:
        """Get all data for an experiment"""
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()

            # Get experiment info
            cursor.execute('SELECT * FROM experiments WHERE id = ?', (experiment_id,))
            experiment = dict(cursor.fetchone() or {})

            # Get data from all module tables
            tables = [
                'm01_environment', 'm02_data_preparation',
                'm03a_qlora_training', 'm03b_grit_training',
                'm04_inference_backends', 'm05_aqi_scores', 'm05_behavioral_metrics',
                'm06_comparative_analysis', 'm07_results_synthesis',
                'model_responses', 'hidden_states'
            ]

            for table in tables:
                if 'experiment_id' in [desc[0] for desc in cursor.execute(f'PRAGMA table_info({table})').fetchall()]:
                    cursor.execute(f'SELECT * FROM {table} WHERE experiment_id = ?', (experiment_id,))
                    experiment[table] = [dict(row) for row in cursor.fetchall()]

            return experiment

    def get_latest_results(self, module: str, limit: int = 10) -> list:
        """Get latest results from a specific module"""
        table_map = {
            'environment': 'm01_environment',
            'data': 'm02_data_preparation',
            'qlora': 'm03a_qlora_training',
            'grit': 'm03b_grit_training',
            'inference': 'm04_inference_backends',
            'aqi': 'm05_aqi_scores',
            'behavioral': 'm05_behavioral_metrics',
            'comparison': 'm06_comparative_analysis',
            'synthesis': 'm07_results_synthesis'
        }

        table = table_map.get(module)
        if not table:
            return []

        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            cursor.execute(f'''
                SELECT * FROM {table}
                ORDER BY created_at DESC
                LIMIT ?
            ''', (limit,))
            return [dict(row) for row in cursor.fetchall()]

    def export_all_data(self, output_path: str):
        """Export all database data to JSON file"""
        import json

        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()

            # Get all table names
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
            tables = [row[0] for row in cursor.fetchall()]

            export_data = {}
            for table in tables:
                cursor.execute(f'SELECT * FROM {table}')
                export_data[table] = [dict(row) for row in cursor.fetchall()]

            with open(output_path, 'w') as f:
                json.dump(export_data, f, indent=2, default=str)

            logging.info(f"Database exported to {output_path}")

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    db = CentralizedDB()
    print("Centralized database initialized successfully")

    # Test experiment creation
    exp_id = db.create_experiment("test_qlora", "qlora", {"lr": 2e-4, "batch_size": 4})
    print(f"Created experiment with ID: {exp_id}")