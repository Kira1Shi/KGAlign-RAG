from model import HuggingFaceChatModel
from prompt_builder import PromptBuilder
from extractor import RelationExtractor



def main():

    text = """
      Elon Musk founded Tesla in 2003 and became its CEO.
      Tesla is an electric vehicle manufacturer headquartered in Austin, Texas.
      The company produces electric cars, battery energy storage systems, and solar products.
      In 2016, Tesla acquired SolarCity, a solar energy services company founded by Lyndon Rive and Peter Rive.
      SpaceX was founded by Elon Musk in 2002 and is headquartered in Hawthorne, California.
      Elon Musk is also the CEO of SpaceX.
      """

    entities = [
          "Elon Musk",
          "Tesla",
          "Austin",
          "Texas",
          "electric cars",
          "battery energy storage systems",
          "solar products",
          "SolarCity",
          "Lyndon Rive",
          "Peter Rive",
          "SpaceX",
          "Hawthorne",
          "California"
      ]

    print("Loading language model...")

    model = HuggingFaceChatModel()

    print("Loading prompt builder...")

    prompt_builder = PromptBuilder(
        template_path="../prompts/pairwise.txt"
    )

    print("Initializing extractor...")

    extractor = RelationExtractor(
        model=model,
        prompt_builder=prompt_builder
    )

    print("Extracting relations...")

    triples = extractor.extract(
        text=text,
        entities=entities
    )

    print("\nExtracted triples:")

    for triple in triples:

        print(triple)



if __name__ == "__main__":

    main()