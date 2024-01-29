from langchain.agents import tool
import wikipedia
import re


def remove_citation_tags(input_string):
    # Use regular expressions to find patterns [cita requerida]
    pattern = re.compile(r'\[cita requerida\]', re.IGNORECASE)

    # Replace the found patterns with an empty string
    result = re.sub(pattern, '', input_string)

    return result


def remove_natural_numbers(input_string):
    # [n]
    pattern = re.compile(r'\[\d+\]')
    result = re.sub(pattern, '', input_string)

    return result


@tool
def get_wikipedia_summary(topic: str) -> str:
    """Busca información en Wikipedia."""

    try:
        info = topic.split(':')
        topic = info[1]
        wiki = wikipedia.set_lang('es')
        wiki = wikipedia.summary(topic, sentences=1)

        wikiResult = remove_natural_numbers(wiki)
        wikiResult = remove_citation_tags(wiki)

        # Verificar si la página existe
        if wiki is not None:
            # Obtener y devolver el resumen de la página
            return wikiResult
        else:
            return "Lo siento, no pude encontrar información sobre ese tema en Wikipedia."

    except:
        return "Lo siento, no pude encontrar información sobre ese tema en Wikipedia."


# # Ejemplo de uso
# topic = "Iphone 14"
# result = get_wikipedia_summary(topic)
# print(result)
