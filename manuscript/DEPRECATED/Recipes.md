# Using Large Language Models to Write Recipes

If you ask the ChatGPT web app to write a recipe using a user supplied ingredient list and a description it does a fairly good job at generating recipes. For the example in this chapter I am taking a different approach:

- Use the recipe and ingredient files from my web app [http://cookingspace.com](http://cookingspace.com) to create context text, given a user prompt for a recipe.
- Treat this as a text prediction problem.
- Format the response for display.

This approach has an advantage (for me!) that the generated recipes will be more similar to the recipes I enjoy cooking since the context data will be derived from my own recipe files.

## Preparing Recipe Data

I am using the JSON Recipe files from my web app [http://cookingspace.com](http://cookingspace.com). The following Python script converts my JSON data to text descriptions, one per file:

```python
import json

def process_json(fpath):
    with open(fpath, 'r') as f:
        data = json.load(f)

    for d in data:
        with open(f"text_data/{d['name']}.txt", 'w') as f:
            f.write("Recipe name: " + d['name'] + '\n\n')
            f.write("Number of servings: " +
                    str(d['num_served']) + '\n\n')
            ingrediants = ["  " + str(ii['amount']) +
                           ' ' + ii['units'] + ' ' +
                           ii['description']
                           for ii in d['ingredients']]
            f.write("Ingredients:\n" +
                    "\n".join(ingrediants) + '\n\n')
            f.write("Directions: " +
                    ' '.join(d['directions']) + '\n')

if __name__ == "__main__":
    process_json('data/vegetarian.json')
    process_json('data/desert.json')
    process_json('data/fish.json')
    process_json('data/meat.json')
    process_json('data/misc.json')
```

Here is a listing of one of the shorter generated recipe files (i.e., text recipe data converted from raw JSON recipe data from my CookingSpace.com web site):

```console
Recipe name: Black Bean Dip

Number of servings: 6

Ingredients:
  2 cup Refried Black Beans
  1/4 cup Sour cream
  1 teaspoon Ground cumin
  1/2 cup Salsa

Directions: Use either a food processor or a mixing bowl and hand mixer to make this appetizer. Blend the black beans and cumin for at least one minute until the mixture is fairly smooth. Stir in salsa and sour cream and lightly mix. Serve immediately or store in the refrigerator.
```

I have generated 41 individual recipe files that will be used for the remainder of this chapter.

In the next section when we use a LLM to generate a recipe, the directions are numbered steps and the formatting is different than my original recipe document files.

## A Prediction Model Using the OpenAI text-embedding-3-large Model

Here we use the **DirectoryLoader** class that we have used in previous examples to load and then create an embedding index.

Here is the listing for the script **recipe_generator.py**:

```python
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.embeddings import OpenAIEmbeddings
from langchain_community.document_loaders import DirectoryLoader
from langchain import OpenAI, VectorDBQA

embeddings = OpenAIEmbeddings(model="text-embedding-3-large")

loader = DirectoryLoader('./text_data/', glob="**/*.txt")
documents = loader.load()
text_splitter = CharacterTextSplitter(chunk_size=2500,
                                      chunk_overlap=0)

texts = text_splitter.split_documents(documents)

docsearch = Chroma.from_documents(texts, embeddings)

qa = VectorDBQA.from_chain_type(llm=OpenAI(temperature=0,
                                model_name=
                                "text-davinci-002"),
                                chain_type="stuff",
                                vectorstore=docsearch)

def query(q):
    print(f"\n\nRecipe creation request: {q}\n")
    print(f"{qa.run(q)}\n\n")

query("Create a new recipe using Broccoli and Chicken")
query("Create a recipe using Beans, Rice, and Chicken")
```

This generated two recipes. Here is the output for the first request:

```console
$ python recipe_generator.py
Running Chroma using direct local API.
Using DuckDB in-memory for database. Data will be transient.

Recipe creation request: Create a new recipe using both Broccoli and Chicken

Recipe name: Broccoli and Chicken Teriyaki
Number of servings: 4

Ingredients:
1 cup broccoli
1 pound chicken meat
2 tablespoons soy sauce
1 tablespoon honey
1 tablespoon vegetable oil
1 clove garlic, minced
1 teaspoon rice vinegar

Directions:

1. In a large bowl, whisk together soy sauce, honey, vegetable oil, garlic, and rice vinegar.
2. Cut the broccoli into small florets. Add the broccoli and chicken to the bowl and toss to coat.
3. Preheat a grill or grill pan over medium-high heat.
4. Grill the chicken and broccoli for 5-7 minutes per side, or until the chicken is cooked through and the broccoli is slightly charred.
5. Serve immediately.
```

If you examine the text recipe files I indexed you see that the prediction model merged information from multiple training data recipes while creating new original text for directions that is loosely based on the directions that I wrote and information encoded in the OpenAI text-davinci-002 model.

Here is the output for the second request:

```console
Recipe creation request: Create a recipe using Beans, Rice, and Chicken

Recipe name: Beans and Rice with Chicken
Number of servings: 4
Ingredients:
1 cup white rice
1 cup black beans
1 chicken breast, cooked and shredded
1/2 teaspoon cumin
1/2 teaspoon chili powder
1/4 teaspoon salt
1/4 teaspoon black pepper
1 tablespoon olive oil
1/2 cup salsa
1/4 cup cilantro, chopped

Directions:
1. Cook rice according to package instructions.
2. In a medium bowl, combine black beans, chicken, cumin, chili powder, salt, and black pepper.
3. Heat olive oil in a large skillet over medium heat. Add the bean mixture and cook until heated through, about 5 minutes.
4. Stir in salsa and cilantro. Serve over cooked rice.
```

## Cooking Recipe Generation Wrap Up

Cooking is one of my favorite activities (in addition to hiking, kayaking, and playing a variety of musical instruments). I originally wrote the [CookingSpace.com](http://cookingspace.com) web app to scratch a personal itch: due to a medical issue I had to closely monitor and regulate my vitamin K intake. I used the US Government's USDA Nutrition Database to estimate the amounts of vitamins and nutrients in some recipes that I use.

When I wanted to experiment with generative models, backed by my personal recipe data, to create recipes, having available recipe data from my previous project as well as tools like OpenAI APIs and LangChain made this experiment simple to set up and run. It is a common theme in this book that it is now relatively easy to create personal projects based on our data and our interests.
