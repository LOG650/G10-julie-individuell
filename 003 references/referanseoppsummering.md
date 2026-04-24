# Referanseoppsummering

Denne filen er laget som et arbeidsnotat for `003 references`. Malet er at du raskt skal kunne bruke den nar du ma svare pa sporsmal om en kilde, forklare hvorfor den er med i oppgaven, eller hente frem hovedinnholdet uten a apne PDF-en pa nytt.

Hver kilde har derfor samme struktur:

- hva kilden handler om
- hva som ble gjort
- hva som ble funnet
- hva du kan bruke den til
- sammendrag av hele kilden

For flere av kildene bygger notatene pa abstract, rapportens egen bruk av kilden, samt offentlig tilgjengelige metadata. Det betyr at dette er detaljerte arbeidsoppsummeringer, men ikke fullverdige litteraturreferat kapittel for kapittel.

## 2026-01+Maritim+verdiskapingsrapport+2026.pdf
**Tittel:** *Maritim verdiskapingsrapport 2026* (Menon Economics, 2026)

- **Hva kilden handler om:** En norsk bransjerapport om storrelse, verdiskaping, sysselsetting og markedsutvikling i maritim naering.
- **Hva som ble gjort:** Rapporten systematiserer naeringstall og utviklingstrekk for ulike deler av maritim sektor, blant annet rederier, verft, utstyr og maritime tjenester.
- **Hva som ble funnet:** Maritim naering har fortsatt stor betydning for norsk verdiskaping. Rapporten viser videre vekst gjennom 2024, men ogsa et marked som er syklisk og avhengig av bredere investerings- og aktivitetsniva.
- **Hva du kan bruke den til:** Som kontekstkilde for hvorfor fartoysutnyttelse, tilgjengelighet og operasjonell robusthet er viktig i maritim sektor. Ikke som metodekilde for forecasting.

### Sammendrag
Denne rapporten er i praksis en kontekstkilde som setter hele prosjektet ditt inn i en norsk maritim ramme. Den gir ikke teori om forecasting og sammenligner heller ikke statistiske modeller med maskinlaering, men den er nyttig fordi den forklarer hvorfor beslutninger rundt drift, utnyttelse og tilgjengelighet faktisk har betydning for en naering som er stor, kapitalintensiv og avhengig av stabile operasjoner. Nar du bruker rapporten, er det derfor ikke fordi den sier noe om hvordan man bygger en SARIMA- eller XGBoost-modell, men fordi den synliggor hvorfor bedre beslutningsstotte i maritim sektor kan ha reell verdi.

Rapporten viser at maritim naering fortsatt er en tung norsk verdiskapingsnaering og at den omfatter flere ulike delsegmenter med ulike markedsbetingelser. Den gir et bilde av en naering som fortsatt har vekst, men der utviklingen ikke er lineer og stabil. Tvert imot er utviklingen knyttet til marked, investeringer og aktivitetsniva, noe som er relevant for en oppgave som handler om forecasting i en volatil offshore-kontekst. I prosjektet ditt kan denne derfor brukes til a begrunne at maritim drift ikke foregar i et stabilt vakuum, men i en naering der bedre prognoser for nedetid eller tilgjengelighet kan ha verdi fordi planlegging, kontrakter og ressursbruk skjer under usikkerhet.

## A Machine Learning based predicitve engine maintenance.pdf
**Tittel:** Ser ut til a vaere *A Machine Learning Based Predictive Maintenance Algorithm for Ship Generator Engines Using Engine Simulations and Collected Ship Data* (Park & Oh, 2023)

- **Hva kilden handler om:** En maritim anvendelse av maskinlaering for prediktivt vedlikehold av skipsgeneratorer.
- **Hva som ble gjort:** Forfatterne kombinerte operative motordata fra skip med simulerte feiltilstander og utviklet et korrigert helsetall for motoren som grunnlag for modellering.
- **Hva som ble funnet:** Studien argumenterer for at mangel pa reelle feildata er et hovedproblem i maritimt prediktivt vedlikehold, og viser at simulerte data kan brukes for a styrke modellene og oppdage avvik tidligere.
- **Hva du kan bruke den til:** Som dokumentasjon pa at AI allerede brukes i maritim drift, spesielt for a redusere uforutsett nedetid.

### Sammendrag
Artikkelen tar utgangspunkt i et praktisk problem som er veldig relevant for maritim drift: det er vanskelig a utvikle gode prediktive vedlikeholdsmodeller fordi ekte feiltilstander er relativt sjeldne, og fordi skip ofte opererer under mange ulike belastninger og omgivelser. Forfatterne forsoker a lose dette ved a kombinere virkelig innsamlede skipdata med simulerte feiltilstander for generator-motorer. Dette er viktig fordi maritim drift i praksis sjelden gir store og ryddige datasett med tydelig merkede feil, og uten et slikt grunnlag blir avanserte modeller ofte svakere enn teorien tilsier.

Det mest sentrale grepet i studien er at forfatterne lager et korrigert helsemal som skal fange opp avvik mer robust enn enkle sensorgrenser alene. Deretter trener de en maskinlaeringsbasert vedlikeholdsalgoritme pa data som inkluderer bade normal drift og simulerte unormale tilstander. Hovedpoenget i artikkelen er ikke bare at modellen fungerer, men at selve dataproblemet ma loses for at maritim AI skal bli nyttig i praksis. Dette gjor kilden relevant i prosjektet ditt fordi den viser at AI i maritim sektor ikke bare handler om a velge en smart modell, men om a ha gode data, riktige representasjoner av problemet og en realistisk kobling mellom modell og operativ bruk. Selv om artikkelen handler om maskineri og ikke om offhire direkte, underbygger den ideen om at datadrevne modeller kan brukes i maritime beslutningsproblemer nar datagrunnlaget er godt nok bygget opp.

## A_Review_of_Artificial_Neural_Networks_Applications_in_Maritime_Industry.pdf
**Tittel:** *A Review of Artificial Neural Networks Applications in Maritime Industry* (Assani et al., 2023)

- **Hva kilden handler om:** En oversiktsartikkel om bruk av kunstige nevrale nettverk i maritim sektor.
- **Hva som ble gjort:** Forfatterne gjennomgikk 69 studier fra det siste ti-aret og grupperte dem etter anvendelse, nettverkstype, treningsmetode og evalueringskriterier.
- **Hva som ble funnet:** ANN brukes bredt til prediksjon, optimalisering, klassifisering og diagnose i maritim sektor, men nytten avhenger sterkt av datakvalitet, modellvalg og validering.
- **Hva du kan bruke den til:** Som bred litteraturkilde for hvorfor LSTM og andre nevrale modeller er relevante i maritim forskning.

### Sammendrag
Denne artikkelen er nyttig fordi den ikke handler om ett enkelt eksperiment, men om hele feltet rundt nevrale nettverk i maritim sektor. Forfatterne samler forskningen fra det siste ti-aret og viser hvordan ANN-metoder har blitt brukt i ulike maritime problem, for eksempel prediksjon, diagnose, klassifisering, optimalisering og beslutningsstotte. Dermed gir artikkelen et oversiktsbilde som er verdifullt i en masteroppgave, fordi den hjelper deg a forklare at nevrale nettverk ikke bare er noe som brukes i generiske data science-prosjekter, men ogsa i maritime anvendelser.

Det viktige funnet i reviewen er ikke at ett bestemt nettverk er best, men at bruken av ANN er bred og voksende, samtidig som kvaliteten pa resultatene avhenger av mer enn bare modelltype. Forfatterne peker pa at datagrunnlag, featurevalg, treningsstrategi og evalueringsmal er avgjorende for om en modell faktisk gir mening. Dette er et viktig poeng for oppgaven din, fordi det passer godt med argumentet om at modellvalg ikke kan vurderes isolert fra datastrukturen. I ditt prosjekt kan denne kilden brukes til a plassere LSTM i en bredere forskningskontekst: LSTM er ikke valgt bare fordi det er avansert, men fordi sekvensmodeller er en etablert del av den maritime AI-litteraturen. Samtidig gir reviewen støtte til en mer nøktern vurdering av slike modeller, siden den viser at praktisk nytte krever gode data og riktig oppsett.

## Application of machine learning techniques for supply chain demand forecasting.pdf
**Tittel:** *Application of Machine Learning Techniques for Supply Chain Demand Forecasting* (Carbonneau, Laframboise & Vahidov, 2008)

- **Hva kilden handler om:** En tidlig sammenligningsstudie mellom ML-modeller og mer tradisjonelle prognosemetoder i supply chain-forecasting.
- **Hva som ble gjort:** Forfatterne testet blant annet nevrale nettverk, rekurrente nettverk og SVM opp mot naive metoder, glidende gjennomsnitt, trend og lineer regresjon pa simulerte og virkelige datasett.
- **Hva som ble funnet:** RNN og SVM gjorde det best i mange settinger, men ikke sa entydig at tradisjonelle eller enklere modeller ble irrelevante.
- **Hva du kan bruke den til:** Som litteraturstotte for at ML kan vaere relevant i forecasting, men ma vurderes empirisk mot enklere alternativer.

### Sammendrag
Denne artikkelen er interessant fordi den er en av de tidligere studiene som forsoker a sammenligne maskinlaering med mer etablerte forecasting-metoder pa en systematisk mate. I stedet for a anta at en avansert modell ma vaere bedre, tester forfatterne flere modellfamilier pa bade simulerte og virkelige data. Det gjor studien relevant for oppgaven din, siden prosjektet ditt ogsa bygger pa ideen om at modellvalg ma avgjores gjennom sammenligning, ikke gjennom antakelser.

Det sentrale bidraget i artikkelen er at den viser hvordan ulike typer modeller oppforer seg under krevende ettersporselsmønstre i supply chain-sammenheng. RNN og SVM presterer sterkt, men studien viser samtidig at en lineer regresjonsmodell fortsatt kan vaere konkurransedyktig. Det betyr at hovedlaerdommen ikke er at ML utkonkurrerer alt annet, men at ML kan gi verdi nar datastrukturen er kompleks, samtidig som mer tradisjonelle metoder ikke bor avskrives. For prosjektet ditt er dette veldig nyttig fordi det støtter en nøktern problemforstaelse: XGBoost og LSTM er relevante kandidater, men de ma sammenlignes med SARIMA og eksponentiell glatting pa det konkrete datasettet. Artikkelen kan dermed brukes som en tidlig litteraturkilde for argumentet om at forecasting ikke er et kapplop mellom "gammelt" og "nytt", men et sporsmal om hva som passer problemstrukturen best.

## Automated machine learning driven model for predicting PSV vessel freight market.pdf
**Tittel:** *Automated Machine Learning Driven Model for Predicting Platform Supply Vessel Freight Market* (Kjeldsberg & Munim, 2024)

- **Hva kilden handler om:** En maritim forecast-studie av PSV-fraktrater med bruk av AutoML.
- **Hva som ble gjort:** Forfatterne brukte 188 manedlige observasjoner og 43 variabler, testet 79 modeller for 1-, 3- og 6-maneders horisonter og sammenlignet dem med triple exponential smoothing.
- **Hva som ble funnet:** Flere datadrevne modeller gjorde det sterkt, blant annet XGBoost-varianter, og historiske rater, riggaktivitet og markedsforhold var viktige drivere.
- **Hva du kan bruke den til:** Som en naer maritim analog til din problemstilling, siden den ogsa kombinerer offshore-kontekst og modellvalg.

### Sammendrag
Denne studien er en av de mest relevante maritime kildene i litteraturlisten din fordi den faktisk handler om forecasting i et offshore-marked, og fordi den sammenligner et bredt sett av modeller i et problem med mange samtidige drivere. Forfatterne ser pa PSV-fraktrater, som er en annen responsvariabel enn offhire, men markedslogikken ligner pa ditt prosjekt ved at den er volatil, avhenger av mange faktorer og ikke uten videre kan forklares av en enkel lineer trend. Derfor er studien et godt eksempel pa hvorfor mer fleksible modeller kan bli relevante i maritim forecasting.

Det som gjor artikkelen sterk, er at den ikke velger en modell pa forhand. I stedet bruker den AutoML og sammenligner et stort sett av kandidater pa flere forecast-horisonter. Resultatet viser at de beste modellene ikke bare er klassiske tidsseriemodeller, men ogsa mer fleksible datadrevne tilnarminger som XGBoost-regresjonstraer og andre maskinlaeringsmodeller. Samtidig identifiserer studien en rekke markedsvariabler som er viktige for prognosen, blant annet tidligere rater, nybyggingspriser, leveringer, ordrebok og riggaktivitet. For prosjektet ditt er dette nyttig fordi det understreker to ting: for det forste at maritim/offshore forecasting ofte er et komplekst problem, og for det andre at gode features og riktig evalueringsoppsett betyr mye. Studien kan derfor brukes som argument for at AI-baserte modeller er faglig relevante i en offshore-kontekst, uten at det betyr at de automatisk vil vinne i enhver case.

## Chen_Guestrin_2016_XGBoost_A_Scalable_Tree_Boosting_System.pdf
**Tittel:** *XGBoost: A Scalable Tree Boosting System* (Chen & Guestrin, 2016)

- **Hva kilden handler om:** Grunnartikkelen for XGBoost.
- **Hva som ble gjort:** Forfatterne formaliserte XGBoost matematisk og beskrev en effektiv implementasjon med regularisering, sparsity-aware split finding og gode skaleringsegenskaper.
- **Hva som ble funnet:** XGBoost viser hvordan tree boosting kan bli bade raskt, robust og skalerbart uten a miste prediksjonsstyrke.
- **Hva du kan bruke den til:** Som den viktigste teorikilden for XGBoost i metode- og teorikapitlet.

### Sammendrag
Denne artikkelen er metodegrunnlaget for XGBoost og er derfor sentral nar du skal forklare hvorfor modellen er med i oppgaven din. Chen og Guestrin presenterer XGBoost som en regularisert gradient boosting-metode basert pa beslutningstraer, men det viktige er at de ikke bare beskriver modellen som en abstrakt algoritme. De viser ogsa hvordan den kan implementeres effektivt, slik at den handterer store datamengder, glisne datasett og praktiske beregningsproblemer pa en mate som gjorde modellen attraktiv i mange anvendelser.

Et hovedpoeng i artikkelen er at XGBoost kombinerer to styrker: pa den ene siden evnen til a fange ikke-lineare sammenhenger og interaksjoner gjennom boosted trees, og pa den andre siden regularisering som hindrer unodig kompliserte modeller. Dette gjør modellen relevant i forecasting-prosjekter der sammenhengene ikke er rent lineere eller der effektene varierer mellom enheter. I din oppgave brukes XGBoost nettopp som en kontrast til SARIMA og eksponentiell glatting: hvis offhire-data best forstas som et feature-basert panelproblem med lagg, rullerende verdier og fartoyspesifikke forskjeller, er XGBoost et naturlig valg. Denne kilden sier ikke at XGBoost alltid er best, men den gir den teoretiske begrunnelsen for hvorfor modellen er sterk og hvorfor den ofte fungerer godt i praksis.

## Commentary on the M5 competition.pdf
**Tittel:** *Commentary on the M5 Forecasting Competition* (Kolassa, 2022)

- **Hva kilden handler om:** En kritisk kommentar til M5-konkurransen og hva den egentlig sier om forecasting.
- **Hva som ble gjort:** Kolassa kommenterer konkurransen med fokus pa enkle modellers rolle, count-data, forklarbarhet og gevinsten av kompleksitet.
- **Hva som ble funnet:** Kommentaren advarer mot a lese M5 som bevis pa at mer kompleks ML alltid er bedre.
- **Hva du kan bruke den til:** Som en nyanserende kilde i diskusjonen om modellkompleksitet og praktisk verdi.

### Sammendrag
Kolassas kommentar er viktig fordi den fungerer som en motvekt til en overfladisk lesning av forecast-konkurranser. M5 blir ofte trukket frem som dokumentasjon pa at avanserte og komplekse modeller gir de beste resultatene. Kolassa viser at dette bildet er for enkelt. Han peker pa at konkurranser av denne typen inneholder spesifikke datatyper, evalueringsmal og strukturer som gjor at resultatene ikke automatisk kan generaliseres til alle forecast-problem.

Artikkelen er derfor nyttig som et kritisk korrektiv. Kolassa understreker at enkle modeller ofte undervurderes, at tellingsdata og hierarkiske serier krever riktig statistisk behandling, og at forklarbarhet fortsatt er viktig i praktisk forecasting. Han argumenterer ogsa for at man ma sporre hva man faktisk far igjen for okt kompleksitet. Dette er veldig relevant i din oppgave, fordi du sammenligner modeller med stor forskjell i kompleksitet og tolkbarhet. Kilden hjelper deg med a argumentere for at lavere feil ikke alene avgjor modellvalg, og at en modell som er marginalt bedre, men mye vanskeligere a tolke eller vedlikeholde, ikke nodvendigvis er det mest forsvarlige valget i praksis.

## Comparing statistical and machine learning methods for time series forecasting.pdf
**Tittel:** *Comparing Statistical and Machine Learning Methods for Time Series Forecasting in Data-Driven Logistics: A Simulation Study* (Schmid et al., 2025)

- **Hva kilden handler om:** En simuleringsstudie om modellvalg i logistiske tidsserier.
- **Hva som ble gjort:** Forfatterne simulerte lineere og ikke-lineare tidsserier og sammenlignet ARIMA, SARIMA, TBATS og trebaserte ML-metoder som Random Forest og XGBoost.
- **Hva som ble funnet:** ML-metodene gjorde det best i de mer komplekse scenariene, mens klassiske modeller fortsatt var sterke nar strukturen var mer regelmessig.
- **Hva du kan bruke den til:** Som en av de mest direkte kildene til argumentet om at modellprestasjon avhenger av datastrukturen.

### Sammendrag
Denne artikkelen er spesielt verdifull fordi den ligner pa problemstillingen i oppgaven din pa et prinsipielt niva. I stedet for a bruke ett enkelt bedriftsdatasett, bygger forfatterne en simuleringsstudie der de kontrollerer hvor lineer eller ikke-lineer datastrukturen er, og deretter ser hvordan ulike modellfamilier presterer. Det gjør studien sterk som litteraturkilde, fordi den tydelig viser at modellvalg ikke bor baseres pa trend eller mote, men pa egenskaper ved problemet.

Resultatene er nyanserte. Trebaserte ML-metoder, og spesielt Random Forest, presterer best i mer komplekse og ikke-lineare scenarier. Samtidig viser studien at klassiske metoder som ARIMA og SARIMA fortsatt er fullt konkurransedyktige nar dataseriene er mer stabile og lettere a modellere med eksplisitt tidsseriedynamikk. Det er nettopp denne typen konklusjon som gjor artikkelen sa nyttig for deg: den stotter ideen om at hverken klassiske modeller eller AI-modeller bor opphoyes til universelle vinnere. I stedet viser den at riktig modell ma velges ut fra hvilke typer mønstre dataene faktisk inneholder. Det gjor den til en sterk kilde bade i litteraturdelen og i diskusjonen av hvorfor resultatene dine ikke automatisk ma favorisere den mest komplekse modellen.

## Enhancing supply chain management.pdf
**Tittel:** Ser ut til a vaere *Enhancing Supply Chain Management: A Comparative Study of Machine Learning Techniques with Cost-Accuracy and ESG-Based Evaluation for Forecasting and Risk Mitigation* (Sattar et al., 2025)

- **Hva kilden handler om:** En sammenligning av flere ML-modeller i supply chain-problemer, vurdert bade pa nøyaktighet, kostnad og ESG.
- **Hva som ble gjort:** Studien brukte modeller som XGBoost og RNN pa prognoser og risikooppgaver, og introduserte egne evalueringsmal for a ta hensyn til mer enn bare ren feil.
- **Hva som ble funnet:** XGBoost var sterk pa prognosenoyaktighet, men enklere modeller kunne komme bedre ut nar kostnad og praktisk anvendbarhet ble tatt med.
- **Hva du kan bruke den til:** Som støtte for at modellvalg i praksis handler om flere kriterier enn bare lavest feilmal.

### Sammendrag
Denne artikkelen er nyttig fordi den flytter oppmerksomheten bort fra en ensidig jakt pa best mulig prediksjonsnoyaktighet. I mange forecast-studier blir modellene vurdert bare pa MAE, RMSE eller lignende mal. Her argumenterer forfatterne for at supply chain-beslutninger ogsa ma vurderes ut fra kostnad, risiko og baerekraft. Derfor introduserer de egne evalueringsmal som forsoker a kombinere prognosenoyaktighet med bredere praktiske hensyn.

Det viktige i studien er ikke bare hvilken modell som vinner, men hvordan svaret endrer seg avhengig av evalueringskriterium. XGBoost gjor det sterkt pa ren nøyaktighet, mens andre modeller kan vaere mer attraktive dersom man ogsa tar hensyn til modellkostnad, kompleksitet eller ESG-perspektiv. Dette er relevant for oppgaven din fordi det hjelper deg med a diskutere forecasting som beslutningsstotte og ikke bare som en teknisk konkurranse. Selv om oppgaven din ikke formaliserer kostnad og ESG pa samme mate, kan kilden brukes til a underbygge at "beste modell" ikke nodvendigvis er den som bare scorer lavest pa ett feilmal. Det passer godt med en diskusjon der tolkbarhet, robusthet og forsvarlighet ogsa betyr noe.

## Friedman_2001_Greedy_Function_Approximation_Gradient_Boosting_Machine.pdf
**Tittel:** *Greedy Function Approximation: A Gradient Boosting Machine* (Friedman, 2001)

- **Hva kilden handler om:** Den klassiske teoriartikkelen bak gradient boosting.
- **Hva som ble gjort:** Friedman formaliserte boosting som gradient descent i funksjonsrom og beskrev generelle loss-funksjoner og boosting med regresjonstraer.
- **Hva som ble funnet:** Artikkelen viser at boosting kan gi sterke og fleksible modeller for bade regresjon og klassifikasjon.
- **Hva du kan bruke den til:** Som teorigrunnlag for boosting og som forløper til XGBoost.

### Sammendrag
Friedmans artikkel er viktig fordi den forklarer gradient boosting pa et teoretisk niva. I stedet for a beskrive boosting som en ren heuristikk, viser han at prosessen kan forstas som numerisk optimering i funksjonsrom. Hver nye modellkomponent legges til for a redusere de feilene som gjenstar etter forrige steg. Denne innsikten er grunnlaget for mye senere utvikling innen tree boosting, inkludert XGBoost.

For prosjektet ditt er artikkelen viktig fordi den hjelper deg med a forklare hvorfor en modell som XGBoost i det hele tatt er en relevant forecasting-modell. Den viser at styrken i boosting ligger i at mange relativt enkle modellkomponenter kan kombineres til en sterk samlet prediktor. Samtidig viser artikkelen at boosting ikke bare handler om a "stappe pa flere traer", men om kontrollert optimering med tapsfunksjoner og regulariseringstankegang. Selv om Friedman ikke skriver om maritim sektor eller forecasting av offhire, gir han det teoretiske grunnlaget for hvorfor boosted trees kan fange ikke-lineare sammenhenger og interaksjoner som klassiske lineere eller univariate modeller ikke lett fanger. Derfor fungerer artikkelen godt som en ren teorikilde i metode- og teorikapitlet.

## Gardner_1985_Exponential_Smoothing_The_State_of_the_Art.pdf
**Tittel:** *Exponential Smoothing: The State of the Art* (Gardner, 1985)

- **Hva kilden handler om:** En klassisk oversiktsartikkel om eksponentiell glatting.
- **Hva som ble gjort:** Gardner oppsummerte utviklingen i feltet og systematiserte Holt-, Winters- og beslektede glattemetoder.
- **Hva som ble funnet:** Eksponentiell glatting framstar som en pragmatisk og ofte sterk metode, men riktig variant avhenger av trend, sesong og datakarakter.
- **Hva du kan bruke den til:** Som en hovedkilde for a begrunne eksponentiell glatting som seriøs prognosemetode og benchmark.

### Sammendrag
Gardners artikkel er en av de viktigste klassiske referansene for eksponentiell glatting. Den fungerer som en statusgjennomgang av feltet frem til midten av 1980-tallet og viser hvordan ulike varianter av metoden kan brukes avhengig av om data har nivå, trend eller sesong. I stedet for a presentere en ny modell isolert, systematiserer artikkelen forskningen og gjor det lettere a forsta hvorfor eksponentiell glatting har blitt en varig del av forecasting-litteraturen.

Det som gjør artikkelen viktig i oppgaven din, er at den understreker at eksponentiell glatting ikke bare er en "enkel baseline". Gardner viser at metoden ofte gir gode prognoser i praksis, nettopp fordi den er fleksibel nok til a handtere ulike typer tidsseriestruktur, samtidig som den er enkel a bruke og tolke. For prosjektet ditt er dette nyttig fordi eksponentiell glatting er en av modellene du sammenligner, og du trenger en solid litteraturkilde som viser at den hører hjemme i en seriøs modellkonkurranse. Artikkelen gir deg grunnlag for a argumentere for at metoden representerer en parsimonisk og veletablert tradisjon i forecasting, ikke bare et enkelt sammenligningspunkt uten teoretisk tyngde.

## Hochreiter_Schmidhuber_1995_Long_Short_Term_Memory_Technical_Report.pdf
**Tittel:** *Long Short-Term Memory* (Technical Report, Hochreiter & Schmidhuber, 1995)

- **Hva kilden handler om:** Den opprinnelige tekniske rapporten som introduserer LSTM.
- **Hva som ble gjort:** Rapporten introduserte minneceller og porter som en ny arkitektur for a handtere lange avhengigheter i rekurrente nettverk.
- **Hva som ble funnet:** Rapporten la grunnlaget for ideen om mer stabil feilflyt og bedre læring over lange sekvenser.
- **Hva du kan bruke den til:** Som utviklingshistorie for LSTM, men normalt er 1997-artikkelen hovedreferansen.

### Sammendrag
Denne tekniske rapporten er den tidlige formuleringen av LSTM-ideen. Hovedproblemet Hochreiter og Schmidhuber adresserer, er at vanlige rekurrente nettverk har vanskeligheter med a lare lange tidsavhengigheter fordi gradienten blir for svak eller ustabil nar informasjon skal propagere over mange tidssteg. Rapporten introduserer derfor minneceller og en mer kontrollert intern struktur som skal bidra til a holde pa relevant informasjon lenger.

Selv om rapporten ikke er den mest brukte akademiske referansen i dag, er den interessant fordi den viser hvordan LSTM ble tenkt fra starten av: som en losning pa et veldig konkret laeringsproblem i sekvensmodeller. For prosjektet ditt er rapporten derfor mest nyttig som bakgrunn eller utviklingshistorie. Den sier noe om hvorfor LSTM ble utviklet og hva slags tidsserieutfordringer modellen er ment a handtere. Dersom du vil forklare hvorfor LSTM prinsipielt kan vaere interessant i offhire-data, er dette en del av svaret: modellen er laget nettopp for sekvenser der viktige mønstre kan ligge langt tilbake i historikken. I den formelle teoridelen er likevel 1997-artikkelen normalt sterkere.

## Hochreiter_Schmidhuber_1997_Long_Short-Term_Memory.pdf
**Tittel:** *Long Short-Term Memory* (Hochreiter & Schmidhuber, 1997)

- **Hva kilden handler om:** Originalartikkelen om LSTM.
- **Hva som ble gjort:** Forfatterne introduserte LSTM som en rekurrent arkitektur med minneceller og porter og testet den pa oppgaver med lange tidsavhengigheter.
- **Hva som ble funnet:** LSTM kunne lare avhengigheter over langt flere tidssteg enn tidligere rekurrente metoder.
- **Hva du kan bruke den til:** Som hovedkilde nar du skal forklare hvorfor LSTM er relevant for tidsserier.

### Sammendrag
Dette er den sentrale originalartikkelen for LSTM og den viktigste referansen din nar du skal forklare modellen teoretisk. Hochreiter og Schmidhuber tar utgangspunkt i problemet med at vanlige rekurrente nettverk ikke klarer a bevare relevant informasjon over lange sekvenser. Nar signalet skal tilbake gjennom mange tidssteg, blir gradienten ofte for liten eller ustabil. LSTM blir presentert som en arkitektur som losser dette ved hjelp av minneceller og porter som regulerer hva som beholdes, oppdateres og sendes videre.

Det viktigste resultatet i artikkelen er at LSTM faktisk kan lare lange avhengigheter betydelig bedre enn tidligere rekurrente tilnarminger. Forfatterne viser at modellen kan handtere sekvenser som strekker seg langt utover det som var praktisk mulig med tidligere metoder. Dette er nettopp grunnen til at LSTM senere ble en standardmodell i mange typer sekvensdata. For prosjektet ditt betyr det at LSTM ikke er tatt med bare fordi det er en kjent deep learning-modell, men fordi den teoretisk er laget for problemer der historiske mønstre kan strekke seg over flere perioder. Samtidig gir artikkelen ogsa indirekte grunnlag for en kritisk diskusjon: bare fordi modellen kan handtere lange avhengigheter, betyr det ikke at ditt datasett faktisk inneholder nok informasjon til at denne fleksibiliteten gir bedre prediksjoner enn enklere modeller.

## Holt_2004_Forecasting_Seasonals_and_Trends.pdf
**Tittel:** *Forecasting Seasonals and Trends by Exponentially Weighted Moving Averages* (Holt, 2004)

- **Hva kilden handler om:** Et klassisk arbeid om hvordan eksponentiell glatting kan utvides til trend og sesong.
- **Hva som ble gjort:** Holt utviklet oppdateringsregler for nivå, trend og sesong med eksponentielt vektede glidende gjennomsnitt.
- **Hva som ble funnet:** Arbeidet la grunnlaget for det som senere ble Holt-Winters-metoder.
- **Hva du kan bruke den til:** Som historisk og teoretisk grunnlag for eksponentiell glatting med trend og sesong.

### Sammendrag
Holts arbeid er sentralt fordi det viser hvordan eksponentiell glatting kan utvides utover helt enkle nivaa-prognoser. I stedet for bare a glatte en serie med tanke pa neste observasjon, utvikler Holt et system der bade trend og sesong kan oppdateres fortlopende. Det betyr at metoden blir relevant for flere typer tidsserier og ikke bare for serier som svinger tilfeldig rundt et konstant nivå.

I prosjektet ditt er kilden viktig fordi den gir de historiske rottene til moderne eksponentiell glatting. Nar du bruker ETS eller relaterte modeller i dag, star du i realiteten pa skuldrene til denne typen klassisk arbeid. Artikkelen viser at prognoser kan bygges opp pa en enkel, rekursiv og operativ mate der nyere observasjoner far storre vekt enn eldre. Dette er et godt motstykke til mer kompliserte modeller som LSTM og XGBoost: eksponentiell glatting representerer en lang tradisjon der mye av styrken ligger i enkelhet, transparens og praktisk nytte. Derfor kan Holt brukes til a underbygge at modellen din ikke bare er en enkel benchmark, men en del av en etablert forecasting-tradisjon.

## Hyndman_Khandakar_2008_Automatic_Time_Series_Forecasting_forecast_Package_R.pdf
**Tittel:** *Automatic Time Series Forecasting: The forecast Package for R* (Hyndman & Khandakar, 2008)

- **Hva kilden handler om:** En metode- og programvareartikkel om automatisk forecasting i R.
- **Hva som ble gjort:** Forfatterne beskrev automatiske algoritmer for bade eksponentiell glatting og ARIMA i `forecast`-pakken.
- **Hva som ble funnet:** Automatisert modellvalg kan gjores metodisk forsvarlig og skalerbart for store mengder tidsserier.
- **Hva du kan bruke den til:** Som praktisk metodekilde for hvordan ARIMA/SARIMA velges og estimeres i praksis.

### Sammendrag
Hyndman og Khandakar er en veldig nyttig kilde fordi den bygger bro mellom teori og praksis. Mange forecasting-artikler beskriver modeller matematisk, men mindre tydelig hvordan modellvalg faktisk blir gjennomfort i reelle systemer. Denne artikkelen viser hvordan det kan gjores automatisk og systematisk i `forecast`-pakken i R, bade for eksponentiell glatting og ARIMA. Dermed blir den en viktig referanse nar du skal forklare at modellvalg kan vaere regelstyrt og transparent, selv om det ikke skjer manuelt for hver enkelt serie.

For prosjektet ditt er ARIMA-delen spesielt viktig. Artikkelen viser hvordan differensiering og modellrangering kan styres ved hjelp av tester og informasjonkriterier, og hvordan en stepwise-sokestrategi kan gi praktisk brukbare modeller uten at man ma identifisere alt fra bunnen av hver gang. Dette er nyttig fordi du i oppgaven din har en praktisk forecasting-pipeline og trenger kilder som forklarer hvorfor en slik framgangsmate er faglig forsvarlig. Kilden gir ogsa en naturlig kobling til hvorfor SARIMA ikke bare er teori, men en modellfamilie som lar seg implementere systematisk pa tvers av flere serier.

## Hyndman_Koehler_Snyder_Grose_2002_State_Space_Exponential_Smoothing.pdf
**Tittel:** *A State Space Framework for Automatic Forecasting Using Exponential Smoothing Methods* (Hyndman et al., 2002)

- **Hva kilden handler om:** En artikkel som setter eksponentiell glatting inn i et state space-rammeverk.
- **Hva som ble gjort:** Forfatterne uttrykte et utvidet sett av glattemetoder som state space-modeller og brukte likelihood, AIC og prediksjonsintervaller.
- **Hva som ble funnet:** Eksponentiell glatting kunne brukes i et mer formelt statistisk oppsett enn tidligere antatt, samtidig som forecast-ytelsen var sterk.
- **Hva du kan bruke den til:** Som hovedkilde for a vise at ETS er mer enn en enkel heuristikk.

### Sammendrag
Denne artikkelen er viktig fordi den formaliserer eksponentiell glatting pa en mate som gjor modellen mer statistisk stringent. Tidligere hadde mange sett pa eksponentiell glatting som en nyttig, men delvis heuristisk metode. Hyndman og medforfatterne viser at en stor familie av slike modeller kan uttrykkes som state space-modeller, noe som gjor det mulig a bruke likelihood-baserte metoder, informasjonkriterier og prediksjonsintervaller pa en mer systematisk mate.

Dette er veldig nyttig for oppgaven din, fordi det gir teoretisk tyngde til eksponentiell glatting som modellfamilie. Nar du sammenligner ETS med SARIMA, XGBoost og LSTM, vil du ikke framstille ETS som en tilfeldig enkel modell, men som en formell og godt etablert forecasting-tilnaerming. Artikkelen hjelper deg med akkurat dette. Den viser at eksponentiell glatting kan brukes automatisk og systematisk, og at forecast-noyaktigheten i mange situasjoner er konkurransedyktig. Dermed stotter den ideen om at det er faglig legitimt a la en slik modell konkurrere med mer komplekse alternativ i en empirisk sammenligning.

## Ljung_Box_1978_Lack_of_Fit_in_Time_Series_Models.pdf
**Tittel:** *On a Measure of Lack of Fit in Time Series Models and Its Applications* (Ljung & Box, 1978)

- **Hva kilden handler om:** Originalkilden til Ljung-Box-testen.
- **Hva som ble gjort:** Forfatterne utviklet et portmanteau-mal for a teste om residualer fortsatt har autokorrelasjon.
- **Hva som ble funnet:** Testen ga et nyttig grunnlag for a vurdere om tidsseriemodeller har fanget opp den relevante tidsstrukturen.
- **Hva du kan bruke den til:** For residualdiagnostikk i SARIMA- og ETS-modeller.

### Sammendrag
Ljung og Box sin artikkel er et klassisk diagnostisk bidrag i tidsserieanalyse. Hovedideen er at en modell ikke bare skal vurderes pa hvor godt den tilpasser observerte data, men ogsa pa om residualene ser tilfeldige ut etter at modellen er estimert. Dersom residualene fortsatt inneholder systematisk autokorrelasjon, tyder det pa at modellen ikke har fanget den underliggende strukturen godt nok. Artikkelen gir dermed et viktig metodisk redskap for kvalitetskontroll av tidsseriemodeller.

I prosjektet ditt er dette relevant fordi residualdiagnostikk er en viktig del av a vurdere SARIMA og til dels eksponentiell glatting. Nar du skriver at residualene ikke viser tydelig gjenværende autokorrelasjon, trenger du en faglig referanse for hvorfor akkurat det betyr noe. Ljung-Box-artikkelen gir dette. Den er ikke en forecasting-artikkel i tradisjonell forstand og sier heller ikke hvilken modell som vil gi lavest prognosefeil. Men den er avgjorende for a vise at tidsseriemodellering ikke bare handler om a "få en prediksjon", men om a kontrollere om modellen faktisk er rimelig spesifisert. Derfor er den særlig nyttig i metode- og verifikasjonsdelen.

## Long Term or Short term predicition of ship detention duration based on machine learning.pdf
**Tittel:** *Long-Term or Short-Term? Prediction of Ship Detention Duration Based on Machine Learning* (Deng & Wan, 2024)

- **Hva kilden handler om:** En maritim ML-studie som klassifiserer varighet av ship detention.
- **Hva som ble gjort:** Forfatterne brukte faktorutvelgelse og random forest for a skille mellom korte og lange detention-perioder for ulike skipstyper.
- **Hva som ble funnet:** Brannsikkerhet, fremdrift/hjelpemaskineri og forurensningsforebygging var sentrale faktorer, og modellen oppnadde relativt god nøyaktighet.
- **Hva du kan bruke den til:** Som en naer maritim eksempelstudie pa operasjonell prediksjon med ML.

### Sammendrag
Denne artikkelen er en god maritim anvendelseskilde fordi den viser hvordan maskinlaering kan brukes til a klassifisere operative maritime hendelser, i dette tilfellet hvor lenge et skip blir tilbakeholdt etter inspeksjon. Selv om detention ikke er det samme som offhire, finnes det en tydelig parallell: begge deler handler om forhold som paverker tilgjengelighet, drift og planlegging, og begge kan sees som beslutningsrelevante hendelser med store praktiske konsekvenser.

Studien bruker en kombinasjon av metode for faktoridentifisering og random forest-klassifikasjon. Det er viktig fordi artikkelen ikke bare trener en modell direkte, men ogsa forsoker a identifisere hvilke typer mangler som faktisk er mest informative for prediksjonen. Resultatene viser at tekniske og sikkerhetsmessige forhold som brannsikkerhet, fremdrift og forurensningsforebygging er blant de viktigste. For prosjektet ditt er denne kilden nyttig fordi den viser at maskinlaering allerede brukes pa maritime operative problem som ligger relativt naert opp til planlegging og risiko. Den kan ogsa brukes til a vise at maritim AI ikke bare handler om marked og vedlikehold, men ogsa om klassifisering av hendelser som har betydning for operasjonell tilgjengelighet.

## M4 competition.pdf
**Tittel:** *The M4 Competition: Results, Findings, Conclusion and Way Forward* (Makridakis, Spiliotis & Assimakopoulos, 2018)

- **Hva kilden handler om:** En stor internasjonal forecast-konkurranse med 100 000 tidsserier.
- **Hva som ble gjort:** M4 sammenlignet statistiske, hybride og ML-baserte forecast-metoder pa tvers av mange serietyper og horisonter.
- **Hva som ble funnet:** Hybride og kombinerte metoder gjorde det best, mens rene ML-metoder generelt ikke dominerte slik mange forventet.
- **Hva du kan bruke den til:** Som empirisk støtte for at forecasting ikke automatisk belonner mest mulig kompleksitet.

### Sammendrag
M4 er en av de viktigste forecast-konkurransene i moderne litteratur fordi den tester modeller i stor skala pa mange forskjellige tidsserier. Poenget med slike konkurranser er ikke bare a rangere modeller, men a se hvilke typer metoder som faktisk leverer robust på tvers av problemer. I M4 var dette spesielt interessant fordi konkurransen inkluderte flere nye og mer avanserte modelltyper enn tidligere, inkludert rene ML-metoder.

Det mest siterte funnet er at hybride og kombinerte metoder gjorde det best, mens rene maskinlaeringsmetoder ikke tok over feltet slik mange hadde ventet. Dette er viktig for oppgaven din fordi det viser at forecasting er et område der klassiske og statistiske komponenter fortsatt står sterkt, selv i moderne benchmark-settinger. Kilden kan brukes til a argumentere for at mer kompleks modellering ikke automatisk gir bedre prognoser, og at den mest forsvarlige tilnarmingen ofte er empirisk benchmarking og nøktern vurdering av datastrukturen. Dette passer godt med prosjektets eget design, der du nettopp sammenligner flere modellfamilier i stedet for a anta at en AI-modell ma vinne.

## M5_accuracy_competition.pdf
**Tittel:** *M5 Accuracy Competition: Results, Findings, and Conclusions* (Makridakis, Spiliotis & Assimakopoulos, 2022)

- **Hva kilden handler om:** Oppsummering av M5-konkurransen pa hierarkiske Walmart-salgsserier.
- **Hva som ble gjort:** Konkurransen krevde prognoser for titusenvis av serier med hierarkisk struktur og rikere datasett enn tidligere M-konkurranser.
- **Hva som ble funnet:** Toppmodellene forbedret benchmarken tydelig, men konkurransen viste ogsa at modellrangering avhenger sterkt av datastruktur, hierarki og features.
- **Hva du kan bruke den til:** Som støtte for at forecasting-resultater ma tolkes i lys av problemoppsett og datastruktur.

### Sammendrag
M5 er viktig fordi den representerer en ny type forecast-konkurranse der datasettet er rikere og mer krevende enn i mange tidligere benchmarker. I stedet for generiske, isolerte tidsserier handler M5 om salgstall med hierarkisk struktur, intermittens og eksogene variabler. Dette gjør konkurransen relevant som referanse for mer komplekse forecasting-problem, men ogsa som en advarsel mot a overfore resultatene ukritisk til andre kontekster.

For prosjektet ditt er M5 nyttig fordi den viser hvor mye datastruktur betyr. Toppmodellene forbedrer benchmarken tydelig, men disse resultatene oppnas i et oppsett der features, hierarki og aggregasjon spiller en stor rolle. Det betyr at sterke resultater i M5 ikke bare handler om "bedre modell", men ogsa om hvordan problemet er representert. Denne innsikten er viktig nar du diskuterer dine egne resultater. Den hjelper deg med a understreke at modellrangering ikke er universell, men avhenger av om data er nulltunge, glisne, hierarkiske eller preget av rike tilleggssignaler. Kilden kan derfor brukes til a nyansere eventuelle påstander om at en modellfamilie generelt er overlegent best.

## Machine Learning and deep learning models .pdf
**Tittel:** Ser ut til a vaere *Machine Learning and Deep Learning Models for Demand Forecasting in Supply Chain Management: A Critical Review* (Douaioui et al., 2024)

- **Hva kilden handler om:** En review av ML- og DL-modeller i demand forecasting.
- **Hva som ble gjort:** Forfatterne gjennomgikk nyere forskning og organiserte litteraturen etter modellfamilier, anvendelser og utvikling over tid.
- **Hva som ble funnet:** AI-baserte prognosemodeller har vokst kraftig i bruk, spesielt der data er komplekse, ikke-lineare eller ustabile.
- **Hva du kan bruke den til:** Som oversiktskilde for hvorfor ML og DL er relevante i moderne forecasting-litteratur.

### Sammendrag
Douaioui og medforfattere gir en bred oversikt over hvordan maskinlaering og dyp laering brukes i ettersporselsprognoser i supply chain management. Artikkelen er nyttig som litteraturkilde fordi den ikke fokuserer pa én enkelt modell, men pa utviklingen i feltet som helhet. Den viser at forecasting-forskning i økende grad inkluderer modeller som XGBoost, LSTM og andre AI-baserte tilnarminger, spesielt nar problemene blir mer komplekse eller mindre godt forklart av klassiske lineere strukturer.

Samtidig er reviewen nyansert. Forfatterne peker ikke bare pa vekst og muligheter, men ogsa pa at valg av modell ma tilpasses problemtype, datakvalitet og praktisk anvendelse. Dette gjor artikkelen nyttig i oppgaven din fordi den stotter at AI-modeller er faglig relevante kandidater, men uten a hevde at de alltid er best. Den kan brukes i litteraturkapitlet for a vise at forecasting-litteraturen har beveget seg mot et bredere metodeutvalg, og i diskusjonen for a underbygge at modellvalg bor handle om problemstruktur fremfor teknologisk mote.

## Post script retail forecasting fildes .pdf
**Tittel:** *Post-Script: Retail Forecasting: Research and Practice* (Fildes, Kolassa & Ma, 2022)

- **Hva kilden handler om:** En oppdatert refleksjon om retail forecasting i lys av nyere forskning og COVID-19.
- **Hva som ble gjort:** Forfatterne tok utgangspunkt i tidligere retail forecasting-litteratur og diskuterte hvordan pandemien og nyere ML-metoder endrer feltet.
- **Hva som ble funnet:** Strukturelle brudd og store sjokk gjør forecasting vanskeligere og krever mer enn enkel videreforing av historiske mønstre.
- **Hva du kan bruke den til:** Som støtte for at forecasting i volatile omgivelser ma tolkes med forsiktighet.

### Sammendrag
Denne artikkelen er ikke en full empirisk studie, men en oppdatert refleksjon over retail forecasting som forsknings- og praksisfelt. Den er skrevet i en periode der pandemien hadde vist hvor raskt markedsmønstre kunne endre seg, og brukes derfor ofte som en kilde til a nyansere forestillingen om at forecasting alltid handler om a ekstrapolere relativt stabile historiske mønstre.

For prosjektet ditt er denne kilden nyttig fordi den hjelper deg med a argumentere for at prognoser i volatile omgivelser ma behandles med forsiktighet. Nar marked eller driftsforhold skifter bratt, blir det vanskeligere for modeller som primart bygger pa historisk mønsterforlengelse. Det betyr ikke at forecasting blir umulig, men at resultatene ma tolkes i lys av usikkerhet, brudd og ekstreme hendelser. Dette poenget passer godt i diskusjonen av offshore- og maritim kontekst, der markedet heller ikke er fullstendig stabilt over tid. Artikkelen kan derfor brukes for a støtte et mer nøkternt syn pa forecast-bruk: modeller gir beslutningsstotte, ikke fasit.

## Towards predictive maintenace in the maritime industry.pdf
**Tittel:** *Towards Predictive Maintenance in the Maritime Industry: A Component-Based Overview* (Kalafatelis et al., 2025)

- **Hva kilden handler om:** En oversiktsartikkel om prediktivt vedlikehold i maritim sektor.
- **Hva som ble gjort:** Forfatterne gjennomgikk forskning etter fartoysystemer og diskuterte blant annet transfer learning, federert laering og implementeringsutfordringer.
- **Hva som ble funnet:** Prediktivt vedlikehold har stort potensial, men datatilgang, forklarbarhet og tillit er fortsatt store barrierer.
- **Hva du kan bruke den til:** Som bred maritim AI-kilde som viser hvordan datadrevne modeller brukes i drift og vedlikehold.

### Sammendrag
Denne artikkelen er nyttig fordi den viser et bredt bilde av hvordan AI og datadrevne modeller brukes i maritime operative problem, spesielt innen prediktivt vedlikehold. I stedet for a fokusere pa én enkelt komponent eller ett skipssystem, organiserer forfatterne litteraturen systematisk etter ulike deler av fartoyet og diskuterer hva slags modeller som brukes, hvilke data som er tilgjengelige og hvilke praktiske hindringer som fortsatt finnes.

Hovedinntrykket i artikkelen er todelt. Pa den ene siden er potensialet stort: bedre prediktivt vedlikehold kan redusere nedetid, forbedre sikkerhet og gi mer effektiv ressursbruk. Pa den andre siden er det mange implementeringsbarrierer, blant annet begrenset datatilgang, behov for forklarbarhet, vanskeligheter med overforing mellom fartoy og behov for tillit hos brukerne. Dette er relevant for oppgaven din fordi det minner om at AI i maritim sektor ikke bare handler om a velge en sterk modell, men om a ha et godt datagrunnlag og en realistisk brukskontekst. Artikkelen kan derfor brukes som bakgrunn for a vise at den maritime sektoren allerede tar i bruk datadrevne metoder, men at praktisk verdi fortsatt avhenger av mer enn ren modellnoyaktighet.

## Vessel Turnaround a machine learning approach.pdf
**Tittel:** *Vessel Turnaround Time Prediction: A Machine Learning Approach* (Chu, Yan & Wang, 2024)

- **Hva kilden handler om:** En XGBoost-basert studie av vessel turnaround time i havn.
- **Hva som ble gjort:** Forfatterne trente en XGBoost-modell pa ankomst- og avgangsdata fra Hong Kong Port og sammenlignet resultatet med rapportert estimated departure time.
- **Hva som ble funnet:** XGBoost ga merkbart lavere MAE og RMSE enn de rapporterte estimatene.
- **Hva du kan bruke den til:** Som en konkret maritim evidenskilde for at XGBoost kan forbedre operative tidsprognoser.

### Sammendrag
Denne studien er nyttig fordi den er en konkret maritim anvendelse av XGBoost i et operativt forecast-problem. Problemstillingen er annerledes enn i prosjektet ditt, siden den handler om vessel turnaround time i havn og ikke om offhire. Likevel er likheten tydelig: begge problemene handler om operasjonell planlegging, tidsavhengighet og behov for mer presise estimater enn det man far gjennom enkle eller manuelle anslag.

Forfatterne bruker data fra Hong Kong Port og trener en XGBoost-regresjonsmodell som sammenlignes med skipenes egne rapporterte estimerte avgangstid. Resultatet er at den datadrevne modellen reduserer feilene tydelig, bade i MAE og RMSE. Det gjor artikkelen til en sterk maritim anvendelseskilde for maskinlaering. I oppgaven din kan den brukes for a vise at XGBoost ikke bare er en generell data science-modell, men en metode som faktisk har vist verdi i maritime operasjonelle prognoser. Samtidig er det viktig a merke seg at problemet i havn ikke er det samme som offhire i offshoreflaten. Kilden er derfor best brukt som analogi og inspirasjon, ikke som direkte bevis for at XGBoost ma vinne i din case.

## What are ARIMA Models_ _ IBM.pdf
**Tittel:** *What Are ARIMA Models?* (IBM)

- **Hva kilden handler om:** En pedagogisk innforing i ARIMA og SARIMA.
- **Hva som ble gjort:** Teksten forklarer Box-Jenkins-logikken, stasjonaritet, differensiering, ACF/PACF og parameterne i ARIMA og SARIMA.
- **Hva som ble funnet:** Dette er en forklaringskilde, ikke en empirisk studie med egne funn.
- **Hva du kan bruke den til:** Som intern huskekilde eller enkel forklaring, men ikke som sterkeste akademiske metodekilde.

### Sammendrag
IBM-teksten er en pedagogisk oversikt over ARIMA-modeller og er mest nyttig som en forklaringskilde. Den gir ikke nye forskningsfunn og er ikke en original metodeartikkel, men den oppsummerer pa en ryddig mate hvordan ARIMA og SARIMA bygges opp, hva parameterne betyr, hvorfor differensiering brukes, og hvordan ACF/PACF inngar i identifisering og diagnostikk. Dette kan vaere nyttig nar du selv trenger en enkel og rask oppfriskning av modellenes logikk.

I oppgaven din er denne kilden mindre viktig enn Hyndman-artiklene eller annen klassisk tidsserielitteratur, men den kan likevel ha verdi som arbeidsstotte. Dersom du skal forklare ARIMA enkelt til noen som ikke kjenner metoden, er den mer tilgjengelig enn mange forskningsartikler. Samtidig bor du vaere forsiktig med a bruke den som hovedkilde i teori- eller metodekapitlet, nettopp fordi den er en populaarfaglig eller forklarende tekst. Den fungerer best som intern orientering og som hjelp til a holde begrepene presise i eget arbeid.

## Winters_1960_Forecasting_Sales_Exponentially_Weighted_Moving_Averages.pdf
**Tittel:** *Forecasting Sales by Exponentially Weighted Moving Averages* (Winters, 1960)

- **Hva kilden handler om:** En klassisk artikkel om forecasting med eksponentielt vektede glidende gjennomsnitt.
- **Hva som ble gjort:** Winters utviklet praktiske forecast-regler for produkter og lager-/produksjonsstyring og viste flere varianter av metoden.
- **Hva som ble funnet:** Metoden ble framstilt som rask, billig og praktisk, og arbeidet ble et viktig grunnlag for Holt-Winters-tradisjonen.
- **Hva du kan bruke den til:** Som historisk hovedkilde for eksponentiell glatting som praktisk forecast-metode.

### Sammendrag
Winters-artikkelen er en historisk klassiker i forecasting-litteraturen og viktig fordi den viser hvordan eksponentielt vektede glidende gjennomsnitt kunne brukes i praktiske forretnings- og styringssituasjoner. Arbeidet ligger tett pa den tradisjonen som senere ble kjent som Holt-Winters-metodene, og det er nyttig fordi det tydelig viser at eksponentiell glatting ikke bare ble utviklet som teoretisk matematikk, men som et operativt forecast-verktøy.

For prosjektet ditt er dette relevant fordi du sammenligner eksponentiell glatting med langt nyere og mer komplekse modeller. Winters hjelper deg med a vise at denne modellfamilien har dype historiske rottter og er bygd for praktisk bruk i serier der man ma oppdatere prognoser fortlopende. Artikkelen understreker at en forecast-metode kan vaere verdifull nettopp fordi den er enkel, rask og robust nok til a brukes pa mange serier. Dette passer godt inn i oppgavens diskusjon av hvorfor enklere modeller fortsatt kan vaere sterke i praksis, selv nar de sammenlignes med ML og deep learning.
