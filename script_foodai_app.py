import customtkinter as ctk
import threading
import requests
import time
import random
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from datetime import datetime

# ==========================================
# CONFIGURAZIONI GLOBALI
# ==========================================
GRAPHDB_URL = "http://localhost:7200/repositories/Tesi" 
GEMINI_API_KEY = ""

# Setup CustomTkinter (Design Moderno)
ctk.set_appearance_mode("Dark")
ctk.set_default_color_theme("green")

class SplashScreen(ctk.CTkToplevel):
    def __init__(self, parent):
        super().__init__(parent)
        self.parent = parent
        
        self.overrideredirect(True)
        width, height = 400, 250
        scale_factor = self._get_window_scaling()
        x = ((self.winfo_screenwidth() / 2) - (width/2)) * scale_factor 
        y = ((self.winfo_screenheight() / 2) - (height/2)) * scale_factor
        self.geometry(f"{width}x{height}+{int(x)}+{int(y)}")
        
        self.grid_columnconfigure(0, weight=1)
        self.grid_rowconfigure((0, 1, 2, 3), weight=1)
        
        self.logo = ctk.CTkLabel(self, text="FoodAI 🧠", font=ctk.CTkFont(size=36, weight="bold"))
        self.logo.grid(row=0, column=0, pady=(40, 10))
        
        self.subtitle = ctk.CTkLabel(self, text="Assistente per la tua Dieta", font=ctk.CTkFont(size=14), text_color="gray")
        self.subtitle.grid(row=1, column=0, pady=(0, 20))
        
        self.progress = ctk.CTkProgressBar(self, mode="indeterminate", width=250)
        self.progress.grid(row=2, column=0, pady=(0, 10))
        self.progress.start()
        
        self.status_label = ctk.CTkLabel(self, text="Inizializzazione sistema...", font=ctk.CTkFont(size=12))
        self.status_label.grid(row=3, column=0, pady=(0, 30))
        self.attributes('-topmost', True)

    def update_status(self, text):
        self.status_label.configure(text=text)

class FoodAIApp(ctk.CTk):
    def __init__(self):
        super().__init__()

        self.withdraw()
        self.splash = SplashScreen(self)

        self.llm_sparql = None
        self.llm_chat = None
        
        threading.Thread(target=self.initialize_app_background, daemon=True).start()

    def initialize_app_background(self):
        try:
            self.splash.update_status("Costruzione dell'Interfaccia Grafica...")
            self.build_ui()
            time.sleep(0.5)

            self.splash.update_status("Inizializzazione Modelli LLM (Gemini)...")
            self.aggiorna_modelli() # Carica i modelli selezionati nei menu a tendina
            time.sleep(1)

            self.splash.update_status("Verifica connessione a GraphDB...")
            try:
                requests.get(GRAPHDB_URL, timeout=3)
                self.splash.update_status("Connessione GraphDB stabilita!")
            except:
                self.splash.update_status("Attenzione: Impossibile connettersi a GraphDB.")
            time.sleep(0.5)
            
            self.after(0, self.show_main_app)
        except Exception as e:
            self.splash.update_status(f"Errore critico: {str(e)}")
            time.sleep(3)
            self.after(0, self.destroy)

    def show_main_app(self):
        self.splash.destroy()
        self.deiconify() 
        self.scrivi_chat("🤖 FoodAI", "Sistema Online e Modelli Inizializzati.\nBenvenuto! Sono il tuo assistente basato su GraphRAG. Quale ricetta o sostituzione alimentare cerchiamo oggi?", "bot")

    def build_ui(self):
        self.title("FoodAI - Assistente per la tua Dieta")
        width, height = 1200, 700
        scale_factor = self._get_window_scaling()
        x = ((self.winfo_screenwidth() / 2) - (width/2)) * scale_factor 
        y = ((self.winfo_screenheight() / 2) - (height/2)) * scale_factor
        self.geometry(f"{width}x{height}+{int(x)}+{int(y)}")
        #self.geometry("1280x720")
        self.grid_columnconfigure(1, weight=1)
        self.grid_rowconfigure(0, weight=1)

        # --- SIDEBAR (Sinistra) ---
        self.sidebar_frame = ctk.CTkFrame(self, width=280, corner_radius=0,fg_color="#0B141A")
        self.sidebar_frame.grid(row=0, column=0, sticky="nsew")
        self.sidebar_frame.grid_rowconfigure(7, weight=1) # Spingo verso il basso

        self.logo_label = ctk.CTkLabel(self.sidebar_frame, text="FoodAI 🧠", font=ctk.CTkFont(size=28, weight="bold"))
        self.logo_label.grid(row=0, column=0, padx=20, pady=(30, 5))
        
        self.subtitle_label = ctk.CTkLabel(self.sidebar_frame, text="Dual-LLM Architecture", font=ctk.CTkFont(size=13), text_color="gray")
        self.subtitle_label.grid(row=1, column=0, padx=20, pady=(0, 30))

        self.info_label = ctk.CTkLabel(self.sidebar_frame, text="📡 STATO SISTEMA:\n🟢 GraphDB (TesiDB)\n🟢 FoodOn Ontology\n🟢 USDA Nutrients\n", justify="left", font=ctk.CTkFont(size=14))
        self.info_label.grid(row=2, column=0, padx=20, pady=0, sticky="w")

        # Dropdown Modello 1 (Architetto)
        self.label_llm1 = ctk.CTkLabel(self.sidebar_frame, text="🧠 LLM 1: Text-to-SPARQL", font=ctk.CTkFont(size=12, weight="bold"))
        self.label_llm1.grid(row=3, column=0, padx=20, pady=(10, 0), sticky="w")
        self.dropdown_llm1 = ctk.CTkOptionMenu(self.sidebar_frame, values=["gemini-3.7-flash","gemini-3.6-flash", "gemini-3.5-flash", "gemini-3.5-flash-lite"], command=lambda _: self.aggiorna_modelli())
        self.dropdown_llm1.set("gemini-3.5-flash-lite") # Usiamo la sintassi corretta delle API correnti
        self.dropdown_llm1.grid(row=4, column=0, padx=20, pady=(5, 10), sticky="ew")

        # Dropdown Modello 2 (Sintetizzatore)
        self.label_llm2 = ctk.CTkLabel(self.sidebar_frame, text="💬 LLM 2: RAG Synthesizer", font=ctk.CTkFont(size=12, weight="bold"))
        self.label_llm2.grid(row=5, column=0, padx=20, pady=(10, 0), sticky="w")
        self.dropdown_llm2 = ctk.CTkOptionMenu(self.sidebar_frame, values=["gemini-3.5-flash-lite", "gemini-3.5-flash",  "gemini-3.6-flash" ], command=lambda _: self.aggiorna_modelli())
        self.dropdown_llm2.set("gemini-3.5-flash-lite") # Sostituto API moderno per flash-lite
        self.dropdown_llm2.grid(row=6, column=0, padx=20, pady=(5, 10), sticky="ew")

        # --- MAIN AREA (Sfondo Verde Smeraldo #008F6F) ---
        self.main_frame = ctk.CTkFrame(self, corner_radius=0, fg_color="#107B64") 
        self.main_frame.grid(row=0, column=1, padx=0, pady=0, sticky="nsew")
        self.main_frame.grid_rowconfigure(0, weight=1)
        self.main_frame.grid_columnconfigure(0, weight=1)

        # Chat Area (Sfondo Whatsapp Chat trasparente per far vedere il verde)
        self.chat_area = ctk.CTkScrollableFrame(self.main_frame, fg_color="#0B141A", corner_radius=10)
        self.chat_area.grid(row=0, column=0, padx=20, pady=(20,0), sticky="nsew")

        # --- INPUT AREA ---
        self.input_frame = ctk.CTkFrame(self.main_frame, fg_color="#202C33", corner_radius=10)
        self.input_frame.grid(row=2, column=0, padx=20, pady=(5,20), sticky="ew")
        self.input_frame.grid_columnconfigure(0, weight=1)

        self.entry_query = ctk.CTkEntry(self.input_frame, placeholder_text="Scrivi un messaggio...", fg_color="#2A3942", border_width=0, height=45, font=ctk.CTkFont(size=15))
        self.entry_query.grid(row=0, column=0, padx=(15, 15), pady=10, sticky="ew")
        self.entry_query.bind("<Return>", lambda event: self.avvia_pipeline())

        self.btn_send = ctk.CTkButton(self.input_frame, text="➤", width=50, fg_color="#00A884", hover_color="#008F6F", command=self.avvia_pipeline, height=45, font=ctk.CTkFont(size=20, weight="bold"))
        self.btn_send.grid(row=0, column=1, padx=(0, 15), pady=10)

    # ==========================================
    # LOGICA DUAL-LLM
    # ==========================================
    def aggiorna_modelli(self):
        """Si attiva all'avvio e quando l'utente cambia opzione nella tendina"""
        modello_sparql = self.dropdown_llm1.get()
        modello_chat = self.dropdown_llm2.get()
        
        # 1. Text-To-SPARQL
        self.llm_sparql = ChatGoogleGenerativeAI(model=modello_sparql, google_api_key=GEMINI_API_KEY)
        
        self.prompt_sparql = PromptTemplate(
            input_variables=["domanda"],
            template="""
            Sei un Data Engineer esperto di SPARQL. Traduci la domanda dell'utente in una query SPARQL.
            STRUTTURA DEL KNOWLEDGE GRAPH:
            - Tipo Ricetta: ?ricetta <http://www.w3.org/1999/02/22-rdf-syntax-ns#type> <http://tesi.unibs.it/ontology#Recipe> .
            - Nome Ricetta: ?ricetta <http://www.w3.org/2000/01/rdf-schema#label> ?nomeRicetta .
            - Nome Ingrediente: ?ingrediente <http://www.w3.org/2000/01/rdf-schema#label> ?nome .
            - Collegamento ricetta-ingrediente: ?ricetta <http://tesi.unibs.it/ontology#usesIngredient> ?ingrediente .

            # --- USDA (Macronutrienti per 100g) - DA USARE SOLO PER INGREDIENTI ---
            ?ingrediente <http://tesi.unibs.it/ontology#hasNutritionProfile> ?usda .
            OPTIONAL {{ ?usda <http://tesi.unibs.it/ontology#hasCalories> ?calorie . }}
            OPTIONAL {{ ?usda <http://tesi.unibs.it/ontology#hasProtein> ?proteine . }}
            OPTIONAL {{ ?usda <http://tesi.unibs.it/ontology#hasFat> ?grassi . }}
            OPTIONAL {{ ?usda <http://tesi.unibs.it/ontology#hasCarb> ?carboidrati . }}

            # --- FoodOn (Famiglie) - DA USARE SOLO PER SOSTITUTI/CATEGORIE ---
            ?ingrediente <http://www.w3.org/1999/02/22-rdf-syntax-ns#type> ?foodon .
            ?foodon <http://www.w3.org/2000/01/rdf-schema#label> ?famiglia .

            REGOLE VITALI:
            1. INIZIO QUERY: NON generare i PREFIX, parti direttamente con SELECT DISTINCT. (Nota: per rdf:type puoi usare la forma estesa <http://www.w3.org/1999/02/22-rdf-syntax-ns#type> oppure la scorciatoia standard 'a').
            2. SCELTA DELLA VARIABILE DA ESTRARRE (MOLTO IMPORTANTE):
            - Se l'utente cerca SOLO UN ELENCO DI RICETTE: scrivi `SELECT DISTINCT ?nomeRicetta`. Nel blocco WHERE usa `?ricetta <http://www.w3.org/1999/02/22-rdf-syntax-ns#type> <http://tesi.unibs.it/ontology#Recipe> .` e `?ricetta <http://tesi.unibs.it/ontology#usesIngredient> [] .` per forzare il database a pescare solo vere ricette e non ingredienti sciolti.
            - Se l'utente cerca INGREDIENTI O SOSTITUTI: scrivi `SELECT DISTINCT ?nome` seguito dai nutrienti.
            3. RICERCA RICETTE CON ESCLUSIONI (es. Senza lattosio, Senza carne, Senza glutine):
            Se l'utente cerca ricette SENZA una categoria di ingredienti (intolleranze, diete o regimi alimentari), sfrutta SEMPRE la gerarchia tassonomica di FoodOn dentro il blocco MINUS. Questo impedisce di escludere ingiustamente i sostituti vegetali (es. permette "almond milk" o "peanut butter" perché non appartengono alla famiglia dei latticini). 
            (Nota: se l'esclusione è invece un singolo ingrediente specifico e non una categoria, es. "senza aglio", puoi usare il controllo lessicale diretto su ?ingrMale <http://www.w3.org/2000/01/rdf-schema#label> ?nomeMale).
            4. RICERCA INGREDIENTI IN UNA RICETTA: Se l'utente cerca una ricetta per nome (es. "spaghetti alla carbonara" o "torta di mele"), traduci il concetto in INGLESE (il database è in inglese) ed estrai solo le 2-3 parole chiave principali. 
            Usa l'AND logico per le parole chiave: FILTER(CONTAINS(LCASE(STR(?nomeRicetta)), "parola1") && CONTAINS(LCASE(STR(?nomeRicetta)), "parola2")). Non cercare mai l'intera frase esatta.
            Se l'utente chiede cosa c'è dentro un piatto specifico (es. "quali sono gli ingredienti della torta di mele"), DEVI estrarre sia la ricetta che i suoi ingredienti concatenati, usando lo stesso schema del punto 2:
            SELECT ?nomeRicetta (GROUP_CONCAT(DISTINCT ?nome; SEPARATOR=", ") AS ?ingredienti)
            E nel blocco WHERE filtra il nome della ricetta: FILTER(CONTAINS(LCASE(STR(?nomeRicetta)), "apple pie"))
            Ricordati sempre di chiudere con GROUP BY ?nomeRicetta LIMIT 50.  
            BIVIO PER L'ESTRAZIONE:
            - CASO A (Solo nomi degli ingredienti): Usa la concatenazione. Scrivi `SELECT ?nomeRicetta (GROUP_CONCAT(DISTINCT ?nome; SEPARATOR=", ") AS ?ingredienti)`. Devi obbligatoriamente aggiungere `GROUP BY ?nomeRicetta` alla fine.
            - CASO B (Nomi + Valori Nutrizionali): Se l'utente vuole i valori nutrizionali degli ingredienti di una ricetta, VIETATO USARE GROUP_CONCAT e GROUP BY. Devi estrarre una riga per ogni ingrediente scrivendo: `SELECT DISTINCT ?nomeRicetta ?nome ?calorie ?proteine ?grassi ?carboidrati`.
            5. RICERCA SOSTITUTI (INGREDIENTI): Se l'utente vuole sostituire l'ingrediente X con la categoria Y, NON cercare X. Cerca SOLO la categoria Y
            (es: se l'utente cerca "dolcificanti/sweetener", il filtro deve essere: FILTER(CONTAINS(LCASE(STR(?famiglia)), "sweetener") || CONTAINS(LCASE(STR(?famiglia)), "sugar substitute") || CONTAINS(LCASE(STR(?nome)), "sweetener") || CONTAINS(LCASE(STR(?nome)), "xylitol") || CONTAINS(LCASE(STR(?nome)), "stevia")). 
            MOLTO IMPORTANTE: Espandi sempre la categoria generica con sinonimi inglesi e nomi specifici. 
            Se l'utente cerca ingredienti o sostituti, parti direttamente da ?ingrediente. VIETATO usare ?ricetta e VIETATO assegnare il tipo Recipe all'ingrediente (gli ingredienti NON sono tesi:Recipe).    
            6. FILTRO ANTI-INDUSTRIALE: Se cerchi ingredienti grezzi, escludi i preparati: FILTER(!CONTAINS(LCASE(STR(?nome)), "mix")).
            7. LIMITI: Aggiungi SEMPRE "LIMIT 50" alla fine della query.
            8. FORMATO OUTPUT: Restituisci SOLO codice SPARQL puro, NESSUN markdown (niente ```sparql).
            9. NUTRIENTI: Includi SEMPRE i nutrienti richiesti dentro i blocchi OPTIONAL {{ ... }} usando due parentesi graffe.
            Domanda: {domanda}
            """
        )
        self.catena_sparql = self.prompt_sparql | self.llm_sparql | StrOutputParser()

        # 2. Sintetizzatore
        self.llm_chat = ChatGoogleGenerativeAI(model=modello_chat, google_api_key=GEMINI_API_KEY)
        
        self.prompt_risposta = PromptTemplate(
            input_variables=["domanda", "dati"],
            template="""
            Sei un assistente culinario basato ESCLUSIVAMENTE sui dati del database aziendale. 
            L'utente ti ha chiesto: "{domanda}"
            
            Dati estratti dal database (reali):
            {dati}
            
            REGOLE ASSOLUTE:
            1. Se i dati dicono "Nessun dato trovato", rispondi: "Mi dispiace, ma non ho trovato informazioni nel mio database riguardo a questa richiesta." Non inventare risposte.
            2. Rispondi in modo cordiale elencando gli ingredienti trovati e i loro valori nutrizionali.
            3. Escludi dalla risposta finale le spezie se l'utente cerca basi proteiche o strutturali.
            4. Se l'utente cerca gli ingredienti di una ricetta i dati potrebbero contenere ingredienti appartenenti a DIVERSE VARIANTI della ricetta cercata. DEVI assolutamente raggruppare gli ingredienti sotto il nome della loro specifica ricetta (campo ?nomeRicetta). NON mischiare tutto in un'unica lista!
            5. NON usare mai frasi introduttive come "Basandomi sul database aziendale...". NON inserire MAI avvisi, disclaimer o "Note aziendali" alla fine del messaggio per giustificare la mancanza di dati. Sii diretto e discorsivo.
            6. Se nei risultati JSON manca un valore nutrizionale richiesto (es. calorie o carboidrati sono nulli/assenti), mostra comunque l'ingrediente nell'elenco, ma scrivi un semplice trattino al posto del numero (es. "Calorie: - "). Non inventare dati e non giustificarti.
            7. Per gli elenchi puntati utilizza • .
            """
        )
        self.catena_risposta = self.prompt_risposta | self.llm_chat | StrOutputParser()

    # ==========================================
    # LOGICA CHAT E PIPELINE
    # ==========================================
    def scrivi_chat(self, mittente, messaggio, tipo):
        row_frame = ctk.CTkFrame(self.chat_area, fg_color="transparent")
        row_frame.pack(fill="x", pady=5)
        
        if tipo == "user":
            bg_color = "#005C4B" 
            text_color = "#E9EDEF"
            ancoraggio = "e" 
            padding_laterale = (100, 10) 
        else:
            bg_color = "#202C33" 
            text_color = "#E9EDEF"
            ancoraggio = "w" 
            padding_laterale = (10, 100) 

        bubble = ctk.CTkFrame(row_frame, fg_color=bg_color, corner_radius=12)
        bubble.pack(anchor=ancoraggio, padx=padding_laterale)

        if tipo == "bot":
            name_label = ctk.CTkLabel(bubble, text=mittente, font=ctk.CTkFont(size=12, weight="bold"), text_color="#1EBEA5")
            name_label.pack(anchor="w", padx=12, pady=(8, 2))

        msg_label = ctk.CTkLabel(bubble, text=messaggio, font=ctk.CTkFont(size=15), text_color=text_color, justify="left", wraplength=600)
        
        pad_y = (2, 10) if tipo == "bot" else (10, 10)
        msg_label.pack(anchor="w", padx=12, pady=pad_y)
        
        self.chat_area.update_idletasks()
        self.chat_area._parent_canvas.yview_moveto(1.0)

    def mostra_ragionamento(self):
        """Crea una bolla temporanea di caricamento nella chat"""
        self.thinking_frame = ctk.CTkFrame(self.chat_area, fg_color="transparent")
        self.thinking_frame.pack(fill="x", pady=5)
        
        bubble = ctk.CTkFrame(self.thinking_frame, fg_color="#202C33", corner_radius=12)
        bubble.pack(anchor="w", padx=(10, 100))
        
        # Testo in corsivo 
        self.thinking_label = ctk.CTkLabel(bubble, text="✨ Analisi della richiesta...", font=ctk.CTkFont(size=14, slant="italic"), text_color="#1EBEA5")
        self.thinking_label.pack(anchor="w", padx=12, pady=10)
        
        self.chat_area.update_idletasks()
        self.chat_area._parent_canvas.yview_moveto(1.0)

    def aggiorna_ragionamento(self, testo):
        """Aggiorna il testo della bolla durante le varie fasi"""
        if hasattr(self, 'thinking_label') and self.thinking_label.winfo_exists():
            self.thinking_label.configure(text=testo)

    def avvia_pipeline(self):
        query_utente = self.entry_query.get().strip()
        if not query_utente:
            return

        self.entry_query.delete(0, 'end')
        self.scrivi_chat("Tu", query_utente, "user")

        self.btn_send.configure(state="disabled")

        self.mostra_ragionamento()

        threading.Thread(target=self.esegui_graphrag, args=(query_utente,), daemon=True).start()

    def esegui_graphrag(self, domanda_utente):
        try:
            self.after(0, self.aggiorna_ragionamento, "🧠 Generazione query SPARQL...")
            query_grezza = self.catena_sparql.invoke({"domanda": domanda_utente}).strip()
            
            prefissi = """
            PREFIX tesi: <http://tesi.unibs.it/ontology#>
            PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
            PREFIX owl: <http://www.w3.org/2002/07/owl#>
            """
            query_finale = prefissi + "\n" + query_grezza

            # Generazione del timestamp corrente
            timestamp = datetime.now().strftime("[%d/%m/%Y - %H:%M:%S]")

            # Stampa formattata
            print(f"\n{timestamp}")
            print(f'Domanda :  "{domanda_utente}"')
            print("Query :")
            print(query_finale)
            print("-" * 50 + "\n")

            self.after(0, self.aggiorna_ragionamento, "📡 Ricerca nel grafo (GraphDB)...")
            headers = {'Accept': 'application/sparql-results+json'}
            risposta_db = requests.get(GRAPHDB_URL, params={'query': query_finale}, headers=headers)
            risposta_db.raise_for_status()
            
            dati_json = risposta_db.json()
            
            ricette_dict = {}
            ingredienti_sciolti = []

            for row in dati_json['results']['bindings']:
                riga = " | ".join([f"{k}: {v['value']}" for k, v in row.items()])
                
                # Se la riga appartiene a una ricetta, viene raggruppata
                if 'nomeRicetta' in row:
                    nome_ricetta = row['nomeRicetta']['value']
                    if nome_ricetta not in ricette_dict:
                        ricette_dict[nome_ricetta] = []
                    ricette_dict[nome_ricetta].append(riga)
                else:
                    ingredienti_sciolti.append(riga)

            risultati_estratti = []

            if ricette_dict:
                nomi_ricette = list(ricette_dict.keys())
                random.shuffle(nomi_ricette)
                ricette_selezionate = nomi_ricette[:10] 
                
                for nome in ricette_selezionate:
                    risultati_estratti.extend(ricette_dict[nome])
            else:
                if len(ingredienti_sciolti) > 25:
                    random.shuffle(ingredienti_sciolti)
                    ingredienti_sciolti = ingredienti_sciolti[:25]
                risultati_estratti = ingredienti_sciolti
              
            contesto_dati = "\n".join(risultati_estratti)
            if not contesto_dati:
                contesto_dati = "Nessun dato trovato nel database."

            self.after(0, self.aggiorna_ragionamento, "📝 Sintesi della risposta in corso...")
            risposta_finale = self.catena_risposta.invoke({
                "domanda": domanda_utente,
                "dati": contesto_dati
            })

            self.after(0, self.completa_pipeline, risposta_finale)

        except requests.exceptions.ConnectionError:
            self.after(0, self.completa_pipeline, f"❌ Errore: Impossibile connettersi a GraphDB all'indirizzo {GRAPHDB_URL}.")
        except Exception as e:
            self.after(0, self.completa_pipeline, f"❌ Errore di sistema: {str(e)}")

    def completa_pipeline(self, risposta_ia):

        if hasattr(self, 'thinking_frame') and self.thinking_frame.winfo_exists():
            self.thinking_frame.destroy() # Distrugge la bolla di caricamento

        self.btn_send.configure(state="normal")

        if hasattr(risposta_ia, 'content'):
            testo_pulito = str(risposta_ia.content)
        else:
            testo_pulito = str(risposta_ia)
            
        self.scrivi_chat("Assistente FoodAI", testo_pulito, "bot")

if __name__ == "__main__":
    app = FoodAIApp()
    app.mainloop()