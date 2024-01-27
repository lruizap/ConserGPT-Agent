from langchain.agents import tool
import wikipedia
import re


def remove_natural_numbers(input_string):
    # [n]
    pattern = re.compile(r'\[\d+\]')
    result = re.sub(pattern, '', input_string)

    return result


@tool
def get_wikipedia_summary(topic: str) -> str:
    """Busca información en Wikipedia."""

    wiki = wikipedia.set_lang('es')
    wiki = wikipedia.summary(topic, sentences=2)

    wikiResult = remove_natural_numbers(wiki)

    # Verificar si la página existe
    if wiki is not None:
        # Obtener y devolver el resumen de la página
        return wikiResult
    else:
        return "Lo siento, no pude encontrar información sobre ese tema en Wikipedia."


# # Ejemplo de uso
# topic = "Inteligencia artificial"
# result = get_wikipedia_summary(topic)
# print(result)
