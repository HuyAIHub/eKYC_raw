import yaml
def get_config():
    with open('config_app/app.yml', encoding='utf-8') as cfgFile:
        config_app = yaml.safe_load(cfgFile)
        cfgFile.close()
    with open('./config_app/ckpt_saved_model.yml', encoding='utf-8') as cfgFile:
        config_model = yaml.safe_load(cfgFile)
        cfgFile.close()
    return config_app, config_model

