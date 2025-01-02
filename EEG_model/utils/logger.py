import csv

class logger:
    def __init__(self, path, initialize):
        self.file = open(path, "w", newline="")
        self.writer = csv.writer(self.file)
        self.writer.writerow(initialize)
        self.file.flush()
    
    def add_row(self, row):
        self.writer.writerow(row)
        self.file.flush()
        
    def close_file(self):
        self.file.close()
        
        
    