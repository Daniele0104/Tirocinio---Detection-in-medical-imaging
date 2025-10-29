Progetto di tirocinio - Daniele Marras 

Un’immagine tratta da campioni di sangue periferico contiene diversi oggetti d’interesse primari, tra cui i globuli rossi, i globuli bianchi e le piastrine. In condizioni anomale, per esempio in presenza di patologie parassitarie quali la malaria, possono presentarsi ulteriori oggetti come i parassiti stessi che tipicamente attaccano i globuli rossi, ma non solo. 

Lo scopo di questo tirocinio è quello di realizzare un tool, in Python, che sia in grado di analizzare delle immagini estratte da campioni di sangue periferico con un duplice scopo: identificare le regioni di interesse (nel caso specifico, dei parassiti che affliggono i globuli rossi) e, successivamente, effettuare un’esaustiva analisi mediante l’utilizzo/addestramento di sistemi basati su tecniche di deep learning end-to-end applicate a parassiti e l’estrazione di feature dalle regioni d’interesse identificate. 

Il tirocinio è suddiviso in due parti. La prima parte consisterà nello studio delle tecniche basilari per poter intraprendere il lavoro vero e proprio e si compone dello studio dal tutorial indicato nel materiale (punto 1), e dell’articolo indicato al punto 2. Dopodiché, per completare la prima parte, verrà effettuato un breve studio, analisi e test del metodo disponibile nel materiale (punto 3): DEIMv2. 

La seconda parte consisterà nell’esecuzione dei seguenti esperimenti: 

Addestramento e valutazione di DEIMv2 sul dataset MP-IDB (Materiale, p4): creazione modello DEIM-MP 

Addestramento e valutazione di DEIMv2 sul dataset IML (Materiale, p4): DEIM-IML 

Addestramento e valutazione di DEIMv2 sul dataset M5 (Materiale, p4): DEIM-M5  

Valutazione di DEIM-MP, DEIM-IML, DEIM-M5 su MalariaBrasil (Materiale, p4)  

Valutazione di DEIM-MP, DEIM-IML, DEIM-M5 sul Dataset MICCAI (inserire link) 

Studio di eventuali strategie di miglioramento dei risultati, come l’applicazione di SAHI in fase di inferenza. 

Eventualmente: studio di migliorie come l’uso di ensemble o l’unione dei dataset in fase di training. 

Materiale: 

Tutorial di Neuromatch (da W1D1 a W2D3) 

Ref1: articolo di riferimento sull’analisi di parassiti di malaria con un sistema di detection (più repository, vari articoli di riferimento presenti all’interno del repository per approfondire) 

Metodo di detection di riferimento	, DEIMv2: Link al repository  

Dataset di riferimento:  

IML 

M5 

MP-IDB 

MalariaBrasil 

NIH-MICCAI (Paper, repository di riferimento) 

Possibile timeline: 

- 50 ore: Tutorial di Neuromatch (da W1D1 a W2D3)  

- 25 ore: Studio e analisi dell’articolo “Ref1” e DEIMv2 

- 125 ore: Sviluppo e implementazione esperimenti seconda parte, punti 1, 2, e 3 

- 100 ore: Sviluppo e implementazione esperimenti seconda parte, punti 4, 5 

- 75 ore: Consolidamento risultati ed eventuali migliorie 

- q.b.: stesura tesi 
