import tensorflow as tf
import zipapp
import requests
import os

logs_directory = 'logs'


class Helper:

    def __init__(self):
        if tf.gfile.Exists(logs_directory) is False:
            tf.gfile.MakeDirs(logs_directory)

        self.dir_count = str(len(tf.gfile.ListDirectory(logs_directory)))

        self.send_zip = 'send_' + self.dir_count + '.zip'

        self.logs_directory = logs_directory + "/" + self.dir_count
        self.model_dir = self.logs_directory + "/model"
        self.output_file = self.logs_directory + "/__main__.py"

        tf.gfile.MakeDirs(self.model_dir)

    def backup(self):
        zipapp.create_archive(self.logs_directory, self.send_zip)

        # then send zip file
        image_filename = os.path.basename(self.send_zip)

        multipart_form_data = {
            'uploaded_file': (image_filename, open(self.send_zip, 'rb')),
        }

        response = requests.post('https://tools.sofora.net/index.php',
                                 files=multipart_form_data)
        print(response.status_code)

    def get_ckpt_dir(self):
        return self.model_dir + "/model"

    def write_file(self, _content):
        _file = open(self.output_file, 'a')
        _file.write(_content)
        _file.close()

