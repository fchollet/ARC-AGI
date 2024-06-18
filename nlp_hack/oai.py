"""Azure OpenAI API"""

import os

from azure.identity import DefaultAzureCredential
from azure.keyvault.secrets import SecretClient
from openai import AzureOpenAI

DEFAULT_KEY_VAULT = "kv-kaiko-main-dev"
API_TYPE = "azure"
API_VERSION = "2023-05-15"
OAI_ENDPOINT = "https://oai--322531.openai.azure.com/"
OAI_SECRET_NAME = "OAI-SECRET"
DEPLOYMENTS = {"gpt-4": "gpt-4-turbo-128k", "gpt-35": "gpt-35-turbo-16k"}

os.environ["AZURE_OPENAI_ENDPOINT"] = OAI_ENDPOINT
os.environ["OPENAI_API_VERSION"] = API_VERSION


def get_secret(secret_name: str) -> str:
    """Get a secret from the lib-nlp Azure Key Vault.

    Args:
        secret_name: Name of the secret to get.

    Returns:
        Value of the secret.
    """
    key_vault_name = os.environ.get("KEY_VAULT_NAME", DEFAULT_KEY_VAULT)
    key_vault_uri = f"https://{key_vault_name}.vault.azure.net"

    credential = DefaultAzureCredential(exclude_shared_token_cache_credential=True)
    client = SecretClient(vault_url=key_vault_uri, credential=credential)

    secret = client.get_secret(secret_name)
    if secret.value is None:
        error_msg = f"Secret {secret_name} not found"
        raise ValueError(error_msg)
    return secret.value


os.environ["AZURE_OPENAI_API_KEY"] = get_secret(OAI_SECRET_NAME)
os.environ["OPENAI_API_KEY"] = get_secret(OAI_SECRET_NAME)


def call_openai_api(message, deployment_name, **kwargs):
    client = AzureOpenAI()
    response = client.chat.completions.create(
        model=deployment_name,
        messages=message,
        **kwargs,
    )
    return response


if __name__ == "__main__":
    response = call_openai_api(
        [{"role": "user", "content": "Hello, how are you?"}],
        DEPLOYMENTS["gpt-35"],
    )
    print(response)
