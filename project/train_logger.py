import sqlite3
import matplotlib.pyplot as plt
import numpy as np
import pathlib

class TrainingLogger:
    def __init__(self, db_path="training_results.db"):
        """
        Initialize the logger with a SQLite database.
        
        Args:
            db_path (str): Path to the SQLite database file.
        """
        if not db_path.endswith('.db'):
            raise ValueError("Database path should end with '.db'")
        if not pathlib.Path(db_path).parent.exists():
            pathlib.Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        if not pathlib.Path(db_path).exists():
            print(f"Creating new database at {db_path}")
        else:
            print(f"Connecting to existing database at {db_path}")
        self.conn = sqlite3.connect(db_path)
        self._create_tables()
        self.conn.execute("PRAGMA synchronous = FULL")

    def _create_tables(self):
        """Create the experiments and results tables if they do not exist."""
        cursor = self.conn.cursor()
        # Table for experiments (each experiment can have multiple runs)
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS experiments (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT,
                description TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        # Table for results (each row is one epoch from one run of an experiment)
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS results (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                experiment_id INTEGER,
                run_id INTEGER,
                epoch INTEGER,
                train_loss REAL,
                train_acc REAL,
                test_loss REAL,
                test_acc REAL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (experiment_id) REFERENCES experiments (id)
            )
        ''')
        self.conn.commit()

    def register_experiment(self, name, description=None):
        """
        Register a new experiment.
        
        Args:
            name (str): Optional name for the experiment.
            description (str): Optional description.
        
        Returns:
            int: The id of the newly registered experiment.
        """
        cursor = self.conn.cursor()
        cursor.execute(
            "SELECT id FROM experiments WHERE name = ?",
            (name,)
        )
        row = cursor.fetchone()
        if row is not None:
            return row[0]



        cursor = self.conn.cursor()
        cursor.execute(
            "INSERT INTO experiments (name, description) VALUES (?, ?)",
            (name, description)
        )
        self.conn.commit()
        return cursor.lastrowid
    
    def get_experiment_ids(self):
        """
        Get the ids of all registered experiments.
        
        Returns:
            list of int: The experiment ids.
        """
        cursor = self.conn.cursor()
        cursor.execute("SELECT id FROM experiments")
        return [row[0] for row in cursor.fetchall()]
    
    def remove_run(self, experiment_id, run_id):
        """
        Remove a run from an experiment.
        
        Args:
            experiment_id (int): The experiment identifier.
            run_id (int): The run identifier.
        """
        self.conn.execute(
            "DELETE FROM results WHERE experiment_id = ? AND run_id = ?",
            (experiment_id, run_id)
        )
        self.conn.commit()
    
    def remove_experiment(self, experiment_id):
        """
        Remove an experiment and all its runs.
        
        Args:
            experiment_id (int): The experiment identifier.
        """
        self.conn.execute(
            "DELETE FROM results WHERE experiment_id = ?",
            (experiment_id,)
        )
        self.conn.execute(
            "DELETE FROM experiments WHERE id = ?",
            (experiment_id,)
        )
        self.conn.commit()
    
    def get_experiment_name(self, experiment_id):
        """
        Get the name of an experiment.
        
        Args:
            experiment_id (int): The experiment identifier.
        
        Returns:
            str: The name of the experiment.
        """
        cursor = self.conn.cursor()
        cursor.execute(
            "SELECT name FROM experiments WHERE id = ?",
            (experiment_id,)
        )
        row = cursor.fetchone()
        return row[0] if row is not None else None

    def get_next_run_id(self, experiment_id):
        """
        Get the next run id for a given experiment.
        
        This is useful when you want to start a new run for the same experiment.
        
        Args:
            experiment_id (int): The experiment identifier.
        
        Returns:
            int: The next run id (starting at 1 if no runs exist).
        """
        cursor = self.conn.cursor()
        cursor.execute(
            "SELECT MAX(run_id) FROM results WHERE experiment_id = ?",
            (experiment_id,)
        )
        row = cursor.fetchone()
        return 1 if row[0] is None else row[0] + 1

    def report_result(self, experiment_id, run_id, epoch, train_loss, train_acc, test_loss, test_acc):
        """
        Log the result for one epoch of a training run.
        
        Args:
            experiment_id (int): The experiment identifier.
            run_id (int): The run identifier for the experiment.
            epoch (int): The epoch number.
            train_loss (float): Training loss.
            train_acc (float): Training accuracy.
            test_loss (float): Test loss.
            test_acc (float): Test accuracy.
        """
        self.conn.execute(
            """
            INSERT INTO results 
                (experiment_id, run_id, epoch, train_loss, train_acc, test_loss, test_acc)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (experiment_id, run_id, epoch, train_loss, train_acc, test_loss, test_acc)
        )
        self.conn.commit()

    def get_results(self, experiment_id, run_id=None):
        """
        Retrieve logged results.
        
        Args:
            experiment_id (int): The experiment identifier.
            run_id (int, optional): If specified, returns results only for that run.
        
        Returns:
            list of tuple: The rows from the results table.
        """
        cursor = self.conn.cursor()
        if run_id is None:
            # Return all results for the experiment (across runs)
            cursor.execute(
                "SELECT run_id, epoch, train_loss, train_acc, test_loss, test_acc "
                "FROM results WHERE experiment_id = ? ORDER BY run_id, epoch",
                (experiment_id,)
            )
        else:
            # Return results for the specified run
            cursor.execute(
                "SELECT epoch, train_loss, train_acc, test_loss, test_acc "
                "FROM results WHERE experiment_id = ? AND run_id = ? ORDER BY epoch",
                (experiment_id, run_id)
            )
        return cursor.fetchall()

    def get_epoch_stats(self, experiment_id, metric='train_loss'):
        """
        Compute the mean and standard deviation for a metric per epoch across all runs.
        
        Args:
            experiment_id (int): The experiment identifier.
            metric (str): The metric to analyze. Should be one of 
                          ['train_loss', 'train_acc', 'test_loss', 'test_acc'].
        
        Returns:
            tuple: Three lists containing epochs, means, and standard deviations.
        """
        cursor = self.conn.cursor()
        cursor.execute(
            f"SELECT epoch, {metric} FROM results WHERE experiment_id = ? ORDER BY epoch",
            (experiment_id,)
        )
        data = cursor.fetchall()
        # Group values by epoch
        epoch_values = {}
        for epoch, value in data:
            epoch_values.setdefault(epoch, []).append(value)
        epochs = sorted(epoch_values.keys())
        means = [np.mean(epoch_values[e]) for e in epochs]
        stds = [np.std(epoch_values[e]) for e in epochs]
        return epochs, means, stds

    def close(self):
        """Close the SQLite database connection."""
        self.conn.close()

def plot_metric(db, *experiment_ids, metric='train_loss', ax=None):
    """
    Plot the mean and standard deviation of a given metric over epochs.
    
    This aggregates the data across all runs of the experiment.
    
    Args:
        experiment_id (int): The experiment identifier.
        metric (str): The metric to plot (e.g., 'train_loss').
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 6))
    for experiment_id in experiment_ids:
        epochs, means, stds = db.get_epoch_stats(experiment_id, metric)
        name = db.get_experiment_name(experiment_id)
            


        ax.plot(epochs, means, label=f'{name} Mean {metric}', marker='o')
        ax.fill_between(epochs,
                        np.array(means) - np.array(stds),
                        np.array(means) + np.array(stds),
                        alpha=0.2, label='Std Dev')
    ax.set_xlabel('Epoch')
    ax.set_ylabel(metric)
    ax.set_title(f'{metric} over Epochs')
    ax.legend()

# Example usage:
if __name__ == '__main__':
    logger = TrainingLogger("test.db")

    # Register a new experiment
    exp_id = logger.register_experiment(name="exp1", description="Testing multiple runs logging.")
    print(f"Registered experiment with id: {exp_id}")

    # Start a new run
    for _ in range(3):
        run_id = logger.get_next_run_id(exp_id)
        print(f"Starting run {run_id} for experiment {exp_id}")

        # Simulate logging results for 5 epochs
        for epoch in range(1, 6):
            # These values would normally come from your training loop.
            train_loss = np.random.random() + 0.5  # dummy value
            train_acc = np.random.random() * 0.1 + 0.8  # dummy value
            test_loss = np.random.random() + 0.6  # dummy value
            test_acc = np.random.random() * 0.1 + 0.75  # dummy value
            logger.report_result(exp_id, run_id, epoch, train_loss, train_acc, test_loss, test_acc)

    # Retrieve and print results
    results = logger.get_results(exp_id)
    print("Logged results:")
    for row in results:
        print(row)

    exp_id_2 = logger.register_experiment(name="exp2", description="Testing multiple runs logging.")
    print(f"Registered experiment with id: {exp_id_2}")
    for _ in range(3):
        run_id_2 = logger.get_next_run_id(exp_id_2)
        print(f"Starting run {run_id_2} for experiment {exp_id_2}")
        
        for epoch in range(1, 6):
            train_loss = np.random.random() + 0.5  # dummy value
            train_acc = np.random.random() * 0.1 + 0.8  # dummy value
            test_loss = np.random.random() + 0.6  # dummy value
            test_acc = np.random.random() * 0.1 + 0.75  # dummy value
            logger.report_result(exp_id_2, run_id_2, epoch, train_loss, train_acc, test_loss, test_acc)

    for id in logger.get_experiment_ids():
        print(f"Experiment {id}: {logger.get_experiment_name(id)}")

    fig, ax = plt.subplots(1, 1, figsize=(20, 10))
    # Plot training loss
    plot_metric(logger, exp_id, exp_id_2, metric='test_acc', ax=ax)
    # Plot test accuracy
    plt.tight_layout()
    plt.savefig("training_results.png")

    # Close the logger when done
    logger.close()
