from agentle.agents.agent import Agent
from agentle.generations.providers.google.google_generation_provider import (
    GoogleGenerationProvider,
)


def listar_chamados_ativos(matricula: str) -> list[str]:
    print(f"Listando chamados para {matricula}")
    return ["Fazer sistema de login", "Criar site no DRUPAL."]


def criar_chamado(matricula: str, descricao: str) -> str:
    print(f"Criando chamado para {matricula} com descrição {descricao}")
    return "Chamado criado com sucesso"


def atender_chamado(matricula: str, chamado: str) -> str:
    print(f"Atendendo chamado para {matricula} com chamado {chamado}")
    return "Chamado atendido com sucesso"


agente_siga = Agent(
    instructions="""Você é um assistente de IA responsável pelo gerenciamento de
    solicitações internas dos Sistemas Integrados da Uniube. Você pode criar chamados (OS)
    criar atendimentos a esses chamados (OS) e listar os chamados existentes.""",
    generation_provider=GoogleGenerationProvider(),
    tools=[listar_chamados_ativos, criar_chamado, atender_chamado],
)

exemplo_resposta = agente_siga.run(
    """
    Matricula: 25962
    Input: Cria uma OS pra mim para um novo sistema de pagamentos que chegou pra eu fazer.
    """
)

print(exemplo_resposta.pretty_formatted())
