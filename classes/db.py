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

    def get_all_table_data(self, tableName, shuffle=False):
        cur = self.connection.cursor()
        retData = []
        for row in cur.execute(f"SELECT * FROM {tableName};"):
            retData.append(row)
        if shuffle: random.shuffle(retData)
        cur.close()
        return retData
    
    def get_all_test_data(self, tableName, shuffle=False):
        cur = self.connection.cursor()
        retData = []
        for row in cur.execute(f"SELECT * FROM {tableName};"):
            retData.append(row)
        if shuffle: random.shuffle(retData)
        cur.close()
        return retData

    def truncate_table(self, tableName):
        cur = self.connection.cursor()
        cur.execute(f"DELETE FROM {tableName};")
        cur.close()

    def get_models_by_acc(self, acc):
        cur = self.connection.cursor()
        retData = []
        for row in cur.execute(f"SELECT m2.*, m.final_acc as first_final_acc FROM models_2 m2 join models m on m2.init_model_id = m.id where m2.final_acc >= {acc} order by id asc"):
            retData.append(row)
        cur.close()
        return retData

    def get_average_best_models(self):
        cur = self.connection.cursor()
        retData = []
        for row in cur.execute("""Select 
            m.id,
            m.layers,
            m.activator,
            m.optimizer,
            m.test_size,
            m.learning_rate,
            (m.max_accuracy  + m2.max_accuracy + m3.max_accuracy) / 3 as av_max_accuracy,
            (m.min_accuracy  + m2.min_accuracy + m3.min_accuracy) / 3 as av_min_accuracy,
            (m.max_loss  + m2.max_loss + m3.max_loss) / 3 as av_max_loss,
            (m.min_loss  + m2.min_loss + m3.min_loss) / 3 as av_min_loss,
            (m.final_acc + m2.final_acc + m3.final_acc) / 3 as av_final_acc
            from models m 
            join models_2 m2 on m2.init_model_id = m.id 
            join models_3 m3 on m3.init_model_id = m.id
            and av_max_accuracy > 0.9
            and av_min_accuracy > 0.45
            order by av_final_acc desc
            """):
            retData.append(row)
        cur.close()
        return retData
    
    def save_model(self, model):
        try:
            cur = self.connection.cursor()
            cur.execute(
                "INSERT INTO models (json_model, layers, activator, optimizer, epochs, max_accuracy, min_accuracy, max_loss, min_loss, final_acc, loss_count, test_size, learning_rate, wrong_ids) VALUES (?, ?, ?, ? ,?, ?, ?, ?, ?, ?, ?, ?, ?, ?)", (model["jsonModel"], model["layers"], model["activator"], model["optimizer"], model["epochs"], model["maxAcc"], model["minAcc"], model["maxLoss"], model["minLoss"], model["finalAcc"], model["lossCount"], model["testSize"], model["learningRate"], model["wrongIds"]))
            self.connection.commit()
            lastId = cur.lastrowid
            cur.close()
            print("Model inserted...")
            return lastId
        except Exception as e:
            print(f"Model insert error: {e}")
    

    def close_connection(self):
        print("Closing DB Connection...")
        self.connection.close()