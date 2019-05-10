import tensorflow as tf
import zipapp

logs_directory = 'logs'
send_zip = 'send.zip'


class Helper:

    def __init__(self):
        if tf.gfile.Exists(logs_directory) is False:
            tf.gfile.MakeDirs(logs_directory)

        self.dir_count = str(len(tf.gfile.ListDirectory(logs_directory)))

        self.logs_directory = logs_directory + "/" + self.dir_count
        self.model_dir = self.logs_directory + "/model"
        self.output_file = self.logs_directory + "/__main__.py"

        tf.gfile.MakeDirs(self.model_dir)

    def backup(self):
        zipapp.create_archive(self.logs_directory, send_zip)

        # then send zip file

    def get_ckpt_dir(self):
        return self.model_dir + "/model"

    def write_file(self, _content):
        _file = open(self.output_file, 'a')
        _file.write(_content)
        _file.close()

