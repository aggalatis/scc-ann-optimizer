import os
import sqlite3
import random

class DB:
    def __init__(self):
        self.DB_FILE = os.getenv('DB_FILE')
        self.connection = None
        self.create_connection()
    
    def create_connection(self):
        try:
            self.connection = sqlite3.connect(self.DB_FILE)
            print("Sqlite connection initialized...")
        except Exception as e:
            print(e)

    def get_all_table_data(self, tableName):
        cur = self.connection.cursor()
        retData = []
        for row in cur.execute(f"SELECT * FROM {tableName};"):
            retData.append(row)
        random.shuffle(retData)
        cur.close()
        return retData

    def truncate_table(self, tableName):
        cur = self.connection.cursor()
        cur.execute(f"DELETE FROM {tableName};")
        cur.close()
    
    def save_model(self, model):
        try:
            cur = self.connection.cursor()
            cur.execute(
                "INSERT INTO models (json_model, layers, activator, optimizer, epochs, max_accuracy, min_accuracy, max_loss, min_loss) VALUES (?, ?, ?, ? ,?, ?, ?, ?, ?)", (model["jsonModel"], model["layers"], model["activator"], model["optimizer"], model["epochs"], model["maxAcc"], model["minAcc"], model["maxLoss"], model["minLoss"]))
            self.connection.commit()
            cur.close()
            print("Model inserted...")
        except Exception as e:
            print(f"Model insert error: {e}")

    def close_connection(self):
        print("Closing DB Connection...")
        self.connection.close()