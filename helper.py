import tensorflow as tf
import zipapp
import requests
import os

logs_directory = 'logs'


class Helper:

    def __init__(self, send_backup=True):
        if tf.gfile.Exists(logs_directory) is False:
            tf.gfile.MakeDirs(logs_directory)

        self.send_backup = send_backup
        self.dir_count = str(len(tf.gfile.ListDirectory(logs_directory)))

        self.send_zip = 'send_' + self.dir_count + '.zip'

        self.logs_directory = logs_directory + "/" + self.dir_count
        self.model_dir = self.logs_directory + "/model"
        self.output_file = self.logs_directory + "/__main__.py"

        tf.gfile.MakeDirs(self.model_dir)

    def backup(self):
        """
        After every model saving, text generated, loss, step, and time log is saved.
        Also with the model. All is zipped and send online for backup.
        :return:
        """
        if self.send_backup:

            # zip backup folder
            zipapp.create_archive(self.logs_directory, self.send_zip)

            # then send zipped folder to the URL
            try:
                requests.post('https://tools.sofora.net/index.php', files={
                    'uploaded_file': (os.path.basename(self.send_zip), open(self.send_zip, 'rb')),
                })
            except requests.exceptions.ConnectionError as error:
                print(error)

    def get_ckpt_dir(self):
        return self.model_dir + "/model"

    def write_file(self, _content):
        _file = open(self.output_file, 'a')
        _file.write(_content)
        _file.close()

