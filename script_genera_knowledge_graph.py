import os
import numpy as np
import pandas as pd
import torch
from owlready2 import get_ontology
from rdflib import Graph, Namespace, URIRef, Literal
from rdflib.namespace import RDF, OWL, RDFS, XSD
import urllib.parse
from tqdm.auto import tqdm
import inflect
import re
from sentence_transformers import SentenceTransformer, util

def main():
    print("\n Avvio ...")
    cartella_corrente = "/content/drive/MyDrive/Tesi/Dataset/"

    p = inflect.engine()

    g = Graph()

    TESI = Namespace("http://tesi.unibs.it/ontology#")
    RECIPE = Namespace("http://tesi.unibs.it/recipe/")
    INGR = Namespace("http://tesi.unibs.it/ingredient/")
    USDA = Namespace("https://fdc.nal.usda.gov/fdc-app.html#/food-details/")

    g.bind("tesi", TESI)
    g.bind("recipe", RECIPE)
    g.bind("ingr", INGR)
    g.bind("usda", USDA)

    print("Caricamento Database USDA e Ontologia FoodOn...")
    df_food = pd.read_csv(os.path.join(cartella_corrente, "food.csv"))
    df_food_nutrient = pd.read_csv(os.path.join(cartella_corrente, "food_nutrient.csv"))
    df_nutrient = pd.read_csv(os.path.join(cartella_corrente, "nutrient.csv"))
    onto = get_ontology(os.path.join(cartella_corrente, "foodon.owl")).load()

    # ==========================================
    # 1. NORMALIZZAZIONE TESTUALE USDA
    # ==========================================
    print("Pulizia e Normalizzazione della nomenclatura USDA...")

    def normalizza_usda(desc: str) -> str:
        if not desc or pd.isna(desc):
            return ""

        testo = str(desc).lower()

        # 1. Rimozione di note amministrative tra parentesi tonde o quadre
        testo = re.sub(r"[\(\[\{].*?[\)\]\}]", "", testo)

        # 2. Rimozione di percentuali e specificatori associati
        testo = re.sub(r"\b\d+(\.\d+)?%\s*\w*", "", testo)

        testo = re.sub(r"\bwith added\b.*?(?=,|$)", "", testo)
        testo = re.sub(r"\byear round average\b", "", testo)
        testo = re.sub(r"\bincludes usda commodity\b.*?(?=,|$)", "", testo)

        testo = re.sub(r"[/;]", " ", testo)
        testo = re.sub(r"[^\w\s,]", "", testo)

        parti = [p.strip() for p in testo.split(",") if p.strip()]

        if not parti:
            return ""

        sostantivo_principale = parti[0]

        if len(parti) == 1:
            return " ".join(sostantivo_principale.split())

        prefissi = []
        suffissi = []

        for elemento in parti[1:]:
            # Categorie merceologiche secondarie e ridondanti da escludere (es. "broiler or fryers")
            if any(scarto in elemento for scarto in ["broiler", "fryer", "not fortified","separable","roaster","select"]):
                continue
            # Specificatori che seguono naturalmente il nome (es. "with skin", "meat only")
            elif elemento.startswith("with ") or "only" in elemento:
                suffissi.append(elemento)
            # Attributi di stato e qualifiche da posporre a sinistra 
            else:
                prefissi.append(elemento)

        # Ricostruzione: [qualificatori/stato] + [sostantivo principale] + [complementi]
        risultato = prefissi + [sostantivo_principale] + suffissi
        stringa_finale = " ".join(risultato)

        return " ".join(stringa_finale.split())

    # Applicazione alla colonna del DataFrame
    df_food['desc_pulita'] = df_food['description'].apply(normalizza_usda)

    # ==========================================
    # IL MODELLO SEMANTICO
    # ==========================================
    print("Caricamento modello di Embedding Neurale (HuggingFace)...")
    modello_semantico = SentenceTransformer('all-MiniLM-L6-v2')
    nomi_usda = df_food['desc_pulita'].tolist()

    print("Calcolo delle coordinate spaziali (Vettori) per database USDA...")
    vettori_usda = modello_semantico.encode(nomi_usda, convert_to_tensor=True, show_progress_bar=True)

    # ==================
    # 2. LETTURA CSV
    # ==================
    percorso_dataset = os.path.join(cartella_corrente, "RecipeNLG_dataset.csv")
    df_ricette = pd.read_csv(percorso_dataset, nrows=25000)

    cache_usda = {}
    cache_foodon = {}

    def converti_in_singolare(termine):

      parole = termine.split()
      if not parole:
        return None

      # Valuta solo l'ultima parola (la testa nominale, es. "red apples" -> "apples")
      singolare_ultima = p.singular_noun(parole[-1])

      if singolare_ultima:
        parole[-1] = singolare_ultima
        return " ".join(parole)
      return None

    # ==========================================
    # 3. SOGLIA ABBASSATA PER I NOMI NORMALIZZATI
    # ==========================================
    SOGLIA_SIMILARITA = 0.60

    print("\nInizio Entity Linking Semantico e Costruzione del Grafo (Macronutrienti)...")

    for index, row in tqdm(df_ricette.iterrows(), total=len(df_ricette), desc="Generazione Triplici"):
        recipe_id = row["Unnamed: 0"]
        nodo_ricetta = URIRef(f"http://tesi.unibs.it/recipe/{recipe_id}")
        g.add((nodo_ricetta, RDF.type, TESI.Recipe))
        titolo_pulito = str(row["title"]).strip()
        g.add((nodo_ricetta, RDFS.label, Literal(titolo_pulito)))

        ingredienti_puliti = eval(row['NER']) if isinstance(row['NER'], str) else row['NER']

        for ing_nome in ingredienti_puliti:
            ing_nome_lower = ing_nome.lower()
            nodo_ingrediente = INGR[urllib.parse.quote(ing_nome.replace(" ", "_"))]

            g.add((nodo_ricetta, TESI.usesIngredient, nodo_ingrediente))
            g.add((nodo_ingrediente, RDFS.label, Literal(ing_nome_lower)))

            # --- ML SEMANTICO per USDA ---
            if ing_nome_lower not in cache_usda:
                vettore_ingrediente = modello_semantico.encode(ing_nome_lower, convert_to_tensor=True)
                similarita = util.cos_sim(vettore_ingrediente, vettori_usda)[0]

                top_candidati_indices = torch.topk(similarita, 10).indices.tolist()
                indice_migliore = top_candidati_indices[0]

                # GUARDRAIL LESSICALE
                parole_alteranti = [
                    "oil", "powder", "extract", "syrup", "baby", "flour", "spray", "spread"
                ]

                for idx in top_candidati_indices:
                    usda_desc = df_food.iloc[idx]['description'].lower()
                    deve_essere_scartato = False
                    for parola in parole_alteranti:
                        if parola in usda_desc.split() and parola not in ing_nome_lower:
                            deve_essere_scartato = True
                            break

                    if not deve_essere_scartato:
                        indice_migliore = idx
                        break

                punteggio_migliore = similarita[indice_migliore].item()

                if punteggio_migliore >= SOGLIA_SIMILARITA:
                    fdc_id = df_food.iloc[indice_migliore]['fdc_id']
                    nutrienti = df_food_nutrient[df_food_nutrient['fdc_id'] == fdc_id]
                    risultato = pd.merge(nutrienti, df_nutrient, left_on='nutrient_id', right_on='id')

                    NUTRIENT_IDS = {
                        'energy': 1008,  # Energy (KCAL) - esclude Atwater e kJ
                        'protein': 1003,  # Protein (G)
                        'fat': 1004,  # Total lipid (fat) (G)
                        'carb': 1005,  # Carbohydrate, by difference (G) - esclude by summation
                    }

                    cal_row = risultato[risultato['nutrient_id'] == NUTRIENT_IDS['energy']].head(1)
                    prot_row = risultato[risultato['nutrient_id'] == NUTRIENT_IDS['protein']].head(1)
                    fat_row = risultato[risultato['nutrient_id'] == NUTRIENT_IDS['fat']].head(1)
                    carb_row = risultato[risultato['nutrient_id'] == NUTRIENT_IDS['carb']].head(1)

                    cache_usda[ing_nome_lower] = {
                        "id": fdc_id,
                        "cal": cal_row.iloc[0]['amount'] if not cal_row.empty else None,
                        "prot": prot_row.iloc[0]['amount'] if not prot_row.empty else None,
                        "fat": fat_row.iloc[0]['amount'] if not fat_row.empty else None,
                        "carb": carb_row.iloc[0]['amount'] if not carb_row.empty else None
                    }
                else:
                    cache_usda[ing_nome_lower] = None

            usda_data = cache_usda[ing_nome_lower]
            if usda_data:
                nodo_usda = USDA[str(usda_data['id'])]
                g.add((nodo_ingrediente, TESI.hasNutritionProfile , nodo_usda))

                if usda_data['cal'] is not None:
                    # Le calorie diventano numeri INTERI (es. 25)
                    g.add((nodo_usda, TESI.hasCalories, Literal(int(round(float(usda_data['cal']))), datatype=XSD.integer)))
                if usda_data['prot'] is not None:
                    # Le proteine diventano DECIMALI arrotondati a 2 cifre (es. 0.60)
                    g.add((nodo_usda, TESI.hasProtein, Literal(round(float(usda_data['prot']), 2), datatype=XSD.decimal)))
                if usda_data['fat'] is not None:
                    # I grassi diventano DECIMALI arrotondati a 2 cifre
                    g.add((nodo_usda, TESI.hasFat, Literal(round(float(usda_data['fat']), 2), datatype=XSD.decimal)))
                if usda_data['carb'] is not None:
                    # I carboidrati diventano DECIMALI arrotondati a 2 cifre
                    g.add((nodo_usda, TESI.hasCarb, Literal(round(float(usda_data['carb']), 2), datatype=XSD.decimal)))

            # --- FOODON (Ontologia) ---

            if ing_nome_lower not in cache_foodon:
              # Tentativo 1: ricerca esatta sul termine così com'è
              risultati_onto = onto.search(label=ing_nome_lower, _case_sensitive=False)

              if risultati_onto:
                cache_foodon[ing_nome_lower] = risultati_onto[0].iri
              else:
                # Tentativo 2: fallback con conversione al singolare (se plurale)
                termine_singolare = converti_in_singolare(ing_nome_lower)

                if termine_singolare:
                  risultati_onto_sing = onto.search(
                      label=termine_singolare, _case_sensitive=False
                  )
                  if risultati_onto_sing:
                    cache_foodon[ing_nome_lower] = risultati_onto_sing[0].iri
                  else:
                    cache_foodon[ing_nome_lower] = None
                else:
                  cache_foodon[ing_nome_lower] = None

            foodon_iri = cache_foodon[ing_nome_lower]
            if foodon_iri:
              g.add((nodo_ingrediente, RDF.type, URIRef(foodon_iri)))

    percorso_salvataggio = os.path.join(cartella_corrente, "FoodKnowledgeGraph.ttl")
    g.serialize(destination=percorso_salvataggio, format="turtle")

    print(f"\nDataset salvato come: FoodKnowledgeGraph.ttl")

if __name__ == "__main__":
    main()