from ddgs import DDGS

def search_text(text, max_results):
    return DDGS().text(text, max_results=max_results)
