from dotenv import dotenv_values


class ConfigProvider:

    def __init__(self, env_path: str):
        self.config = dotenv_values(env_path)

    def get_env(self, key: str):
        value = self.config.get(key)
        if value is None:
            raise ValueError(f"Environment variable '{key}' not defined")
        return value

    def get_config(self):
        return self.config


class AppConfig:

    def __init__(self, config_provider: ConfigProvider):
        self.OPENAI_API_KEY = config_provider.get_env('OPENAI_API_KEY')
