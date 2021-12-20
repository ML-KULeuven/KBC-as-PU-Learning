import os

__project_info_path = os.path.dirname(os.path.realpath(__file__))
project_dir = os.path.dirname(__project_info_path)
data_dir = os.path.join(project_dir, 'data')
amie_jar_settings_json = os.path.join(project_dir, 'amie_dir.json')
