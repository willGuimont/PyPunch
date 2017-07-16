

class DataSaver:
    @staticmethod
    def save_to_file(data, path):
        with open(path, 'w') as f:
            for (x, y) in data:
                f.write('{x},{y}\n'.format(x=x, y=y))

if __name__ == '__main__':
    pass
