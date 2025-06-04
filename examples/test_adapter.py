import logging
from agentle.generations.tools.tool import Tool
from agentle.generations.providers.google.adapters.agentle_tool_to_google_tool_adapter import (
    AgentleToolToGoogleToolAdapter,
)

# Configure logging to show all debug logs
logging.basicConfig(
    level=logging.DEBUG, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)


def listar_chamados_ativos(matricula: str) -> list[str]:
    """Lista os chamados ativos para uma matrícula."""
    print(f"Listando chamados para {matricula}")
    return ["Fazer sistema de login", "Criar site no DRUPAL."]


def criar_chamado(matricula: str, descricao: str) -> str:
    """Cria um novo chamado para uma matrícula."""
    print(f"Criando chamado para {matricula} com descrição {descricao}")
    return "Chamado criado com sucesso"


def atender_chamado(matricula: str, chamado: str) -> str:
    """Atende um chamado específico."""
    print(f"Atendendo chamado para {matricula} com chamado {chamado}")
    return "Chamado atendido com sucesso"


def test_adapter():
    """Test the adapter with various tools."""
    adapter = AgentleToolToGoogleToolAdapter()

    # Create tools from functions
    tools = [
        Tool.from_callable(listar_chamados_ativos),
        Tool.from_callable(criar_chamado),
        Tool.from_callable(atender_chamado),
    ]

    print("=" * 80)
    print("TESTING AGENTLE TOOL TO GOOGLE TOOL ADAPTER")
    print("=" * 80)

    # Test each tool
    for i, tool in enumerate(tools, 1):
        print(f"\n--- Testing Tool {i}: {tool.name} ---")
        print(f"Original Agentle Tool parameters: {tool.parameters}")

        try:
            google_tool = adapter.adapt(tool)
            print(f"✅ Successfully converted '{tool.name}'")

            # Show the function declaration details
            if google_tool.function_declarations:
                func_decl = google_tool.function_declarations[0]
                print(f"Function name: {func_decl.name}")
                print(f"Description: {func_decl.description}")

                if func_decl.parameters and func_decl.parameters.properties:
                    properties = func_decl.parameters.properties
                    print(f"Parameters ({len(properties)}):")
                    for param_name, param_schema in properties.items():
                        print(f"  - {param_name}: {param_schema.type}")
                    print(f"Required: {func_decl.parameters.required or []}")
                else:
                    print("No parameters")
            else:
                print("No function declarations found")

        except Exception as e:
            print(f"❌ Failed to convert '{tool.name}': {e}")

        print("-" * 50)

    print("\n🎉 Adapter testing completed!")


if __name__ == "__main__":
    test_adapter()
