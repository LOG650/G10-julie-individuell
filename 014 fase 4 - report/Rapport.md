::: {custom-style="Title"}
Prognostisering av offhire for fartøy i offshoresegmentet
:::

::: {custom-style="Subtitle"}
En sammenligning av SARIMA, Eksponentiell Glatting, XGBoost og LSTM
:::

**Forfatter(e):** Julie Bjørheim

**Totalt antall sider inkludert forsiden:** 97

**Molde, innleveringsdato:** 31.05.2026

---

**Obligatorisk egenerklæring/gruppeerklæring**

Den enkelte student er selv ansvarlig for å sette seg inn i hva som er lovlige hjelpemidler, retningslinjer for bruk av disse og regler om kildebruk. Erklæringen skal bevisstgjøre studentene på deres ansvar og hvilke konsekvenser fusk kan medføre. Manglende erklæring fritar ikke studentene fra sitt ansvar.

Du fyller ut erklæringen ved å klikke i ruten til høyre for den enkelte del `1`–`6` i Word-versjonen.

1. Jeg erklærer herved at min besvarelse er mitt eget arbeid, og at jeg ikke har brukt andre kilder eller mottatt annen hjelp enn det som er nevnt i besvarelsen.
2. Jeg erklærer videre at denne besvarelsen:
   - ikke har vært brukt til annen eksamen ved annen avdeling, universitet eller høgskole innenlands eller utenlands
   - ikke refererer til andres arbeid uten at det er oppgitt
   - ikke refererer til eget tidligere arbeid uten at det er oppgitt
   - har alle referansene oppgitt i litteraturlisten
   - ikke er en kopi, duplikat eller avskrift av andres arbeid eller besvarelse
3. Jeg er kjent med at brudd på ovennevnte er å betrakte som fusk og kan medføre annullering av eksamen og utestengelse fra universiteter og høgskoler i Norge.
4. Jeg er kjent med at alle innleverte oppgaver kan bli plagiatkontrollert.
5. Jeg er kjent med at høgskolen vil behandle alle saker hvor det foreligger mistanke om fusk etter høgskolens retningslinjer.
6. Jeg har satt meg inn i regler og retningslinjer for bruk av kilder og referanser.

**Personvern**

**Personopplysningsloven**

Forskningsprosjekt som innebærer behandling av personopplysninger i henhold til personopplysningsloven skal meldes til Norsk senter for forskningsdata, NSD, for vurdering.

- **Har oppgaven vært vurdert av NSD?** Nei
- **Referansenummer:** N/A
- **Jeg erklærer at oppgaven ikke omfattes av personopplysningsloven.**

**Helseforskningsloven**

Dersom prosjektet faller inn under helseforskningsloven, skal det også søkes om forhåndsgodkjenning fra Regionale komiteer for medisinsk og helsefaglig forskningsetikk, REK.

- **Har oppgaven vært til behandling hos REK?** Nei
- **Referansenummer:** N/A

**Publiseringsavtale**

- **Studiepoeng:** 15
- **Veileder:** Per Kristian Rekdal, Bård Inge Austigaard Pettersen
- **Elektronisk publisering:** Ja
- **Båndlagt (konfidensiell):** Nei
- **Publisering etter båndleggingsperiode:** N/A
- **Dato:** 31.05.2026
- **Antall ord:** 18709
- **Forfattererklæring:** [fylles inn hvis påkrevd]

**KI-erklæring**

Erklæring om bruk av kunstig intelligens (KI) på hjemmeeksamen:

- **Har du benyttet KI-verktøy i din besvarelse?** Ja
- **Tekstgenerering og skrivehjelp:** Ja
- **Språkvask og korrektur:** Ja
- **Programmering og kodehjelp:** Ja
- **Hjelp til å analysere digitale data:** Ja
- **Lage bilder og figurer:** Ja
- **Annet:** Ja
- Jeg bekrefter at detaljert forklaring av hvilket KI-verktøy som er brukt, og hvordan det er brukt, er lagt inn i oppgaven under overskriften `Bruk av kunstig intelligens`.
- Jeg bekrefter at all bruk av KI-verktøy i min hjemmeeksamen eller oppgave er beskrevet i besvarelsen.
- Jeg er kjent med retningslinjene for bruk av KI på hjemmeeksamen ved Høgskolen i Molde.
- Teksten som er levert inn er min egen, uavhengig av KI-verktøy.

Julie Bjørheim, 31.05.2026

---

```{=openxml}
<w:p><w:r><w:br w:type="page"/></w:r></w:p>
```

<div class="pdf-page-break"></div>

**Forord**

Denne oppgaven er skrevet som en del av emnet LOG650 Forskningsprosjekt ved Høgskolen i Molde. Temaet for oppgaven er prognostisering av månedlig offhire for fartøy i offshoresegmentet, med særlig vekt på sammenligning av tradisjonelle tidsseriemodeller og KI baserte prognosemodeller. Gjennom prosjektet har jeg fått bedre innsikt i hvordan historiske driftsdata kan brukes som beslutningsstøtte, men også hvilke begrensninger som oppstår når datagrunnlaget er kort, ujevnt og preget av mange nullobservasjoner.

Jeg ønsker å rette en takk til Simon Møkster Shipping AS for tilgang til datagrunnlaget. Jeg vil også takke veilederne ved Høgskolen i Molde for faglige innspill og støtte underveis i arbeidet. Generative KI verktøy er brukt som støtte i arbeidsprosessen, blant annet til ideutvikling, strukturering, språklig bearbeiding og kodearbeid. Alle faglige vurderinger, modellvalg, analyser og endelige formuleringer er gjennomgått og kvalitetssikre.

```{=openxml}
<w:p><w:r><w:br w:type="page"/></w:r></w:p>
```

<div class="pdf-page-break"></div>

**Sammendrag**

Denne oppgaven undersøker hvordan valg av prognosemodell påvirker prediksjonsnøyaktigheten for månedlig offhire for fartøy innenfor samme offshoresegment. I studien betegnes offhire som prosentandel dager per måned registrert uten kontrakt og det blir brukt som en indikator på redusert kontraktsutnyttelse. Studien er gjennomført som en kvantitativ, casebasert sammenligning av fire prognosemodeller: SARIMA, eksponentiell glatting, XGBoost og LSTM.

Datagrunnlaget består av historiske, anonymiserte offhire-data fra Simon Møkster Shipping AS for 16 fartøy. Den direkte historiske modellsammenligningen bygger på 15 fartøy, fordi ett fartøy manglet tilstrekkelig historikk for lik evaluering. Modellene ble estimert og evaluert innenfor samme historiske oppsett, med eksplisitt trenings- og testdeling og ekspanderende én-stegs prognoser gjennom testperioden. Prediksjonsnøyaktigheten ble vurdert ved hjelp av MAE, RMSE, sMAPE og MASE.

Resultatene viser at modellvalg har betydning for prediksjonsnøyaktigheten, men ikke på en måte som automatisk gir fordel til de mest komplekse modellene. SARIMA oppnådde lavest MAE, RMSE og MASE i den historiske testen, og fremstår derfor som det mest forsvarlige førstevalget i denne casen. XGBoost og LSTM var konkurransedyktige på absolutte feil, men overgikk ikke SARIMA samlet sett. Eksponentiell glatting fungerte som en nyttig referansemodell, men kom svakere ut på MAE og RMSE.

Fremtidsprognosene for 1, 3, 6 og 12 måneder frem viste samtidig at modellene ga ulike fremtidsbilder. Forskjellene økte særlig på lengre horisonter, og XGBoost ga et markant høyere prognosenivå enn de øvrige modellene på 12 måneders horisont. Dette understreker at lange fler-stegsprognoser bør tolkes som usikre modellbaner eller scenario indikasjoner, ikke som presise forventninger. Studien konkluderer med at historiske offhire-data kan ha praktisk verdi som beslutningsstøtte, men at prognosene bør brukes med skjønn og ses i sammenheng med operasjonell kunnskap, kontraktsforhold og markedskontekst.

```{=openxml}
<w:p><w:r><w:br w:type="page"/></w:r></w:p>
```

<div class="pdf-page-break"></div>

**Abstract**

This thesis examines how the choice of forecasting model affects prediction accuracy for monthly vessel offhire within the same offshore segment. In the study, offhire is operationalized as the monthly percentage of days registered without contract, used as an indicator of reduced contract utilization. The study is designed as a quantitative, case-based comparison of four forecasting models: SARIMA, exponential smoothing, XGBoost, and LSTM.

The empirical basis consists of historical, anonymized offhire data from Simon Møkster Shipping AS for 16 vessels. The direct historical model comparison is based on 15 vessels, because one vessel lacked sufficient history for like-for-like evaluation. All models were estimated and evaluated under the same historical setup, using an explicit train/test split and expanding one-step-ahead forecasts throughout the test period. Predictive accuracy was assessed using MAE, RMSE, sMAPE, and MASE.

The results show that model choice affects predictive accuracy, but not in a way that automatically favors the most complex models. SARIMA achieved the lowest MAE, RMSE, and MASE in the historical test, and therefore appears to be the most defensible first choice in this case. XGBoost and LSTM were competitive in terms of absolute errors, but did not outperform SARIMA overall. Exponential smoothing served as a useful reference model, but performed weaker on MAE and RMSE.

The future forecasts for 1, 3, 6, and 12 months ahead also showed that the models produced different future paths. The differences became particularly clear at longer horizons, where XGBoost produced substantially higher forecast levels than the other models at the 12-month horizon. This highlights that long recursive multi-step forecasts should be interpreted as uncertain model paths or scenario-like indications, rather than precise expectations. The study concludes that historical offhire data can provide practical value as decision support, but that forecasts should be interpreted with professional judgment and considered together with operational knowledge, contract conditions, and market context.

```{=openxml}
<w:p><w:r><w:br w:type="page"/></w:r></w:p>
```

<div class="pdf-page-break"></div>

::: {#toc-placeholder}
:::

```{=openxml}
<w:p><w:r><w:br w:type="page"/></w:r></w:p>
```

<div class="pdf-page-break"></div>

# Innledning

Fartøy som opererer i offshoresegmentet, er en sentral del av infrastrukturen rundt petroleumsaktiviteten på norsk sokkel. De transporterer utstyr, drivstoff, forsyninger og andre objekter som er nødvendige for å opprettholde aktiviteten. Samtidig foregår operasjonene i et miljø preget av teknisk kompleksitet, kontraktsmessige forpliktelser og krevende operasjonelle rammevilkår. I denne konteksten kan perioder med offhire, få betydelige konsekvenser både for rederiene og for oppdragsgiverne.

For rederier innebærer offhire ikke bare tapte inntekter, men også økt usikkerhet knyttet til planlegging, ressursutnyttelse, kontraktsoppfølging og konkurranseevne. For operatørene kan samme offhire skape forstyrrelser i logistikk og drift. Behovet for mer presise prognoser blir særlig tydelig i et marked som samtidig er preget av betydelige svingninger i aktivitet, investeringsnivå og fartøyrater (Menon Economics, 2026). Bedre beslutningsstøtte for offhire er derfor ikke bare et teknisk analyseproblem, men også et praktisk spørsmål om operasjonell robusthet i et volatilt marked.

Tradisjonelt har prognostisering innen logistikk og operasjonsstyring i stor grad vært basert på klassiske statistiske modeller, særlig ARIMA modeller og eksponentiell glatting. Disse modellene er metodisk etablerte og egner seg godt når historiske tidsserier inneholder relativt stabile mønstre i nivå, trend og sesong (Gardner, 1985; Hyndman, Koehler, Snyder, & Grose, 2002; Hyndman & Khandakar, Automatic time series forecasting: The forecast package for R, 2008). Samtidig viser nyere forskning at maskinlæringsmodeller kan være konkurransedyktige eller bedre når datastrukturen er mer kompleks, heterogen eller ikke-lineær (Carbonneau, Laframboise, & Vahidov, 2008; Schmid, Roidl, Kirchheim, & Pauly, 2025).

Innen maritim sektor har maskinlæringsmodeller de siste årene særlig vært brukt til prediktivt vedlikehold, markedsprediksjoner og andre operasjonelle beslutningsproblemer. Forskning som direkte sammenligner tradisjonelle tidsseriemodeller og KI-baserte modeller for prognostisering av offhire på fartøynivå er derimot mer begrenset. Dette gjør det relevant å undersøke hvordan ulike modellfamilier presterer i akkurat denne konteksten (Chu, Yan, & Wang, 2024; Kalafatelis, et al., 2025; Kjeldsberg & Munim, 2024).

## Problemstilling

Formålet med denne studien er å undersøke hvordan modellvalg påvirker prognostisering av månedlig offhire i offshorenæringen. På bakgrunn av behovet for mer presis beslutningsstøtte i et volatilt marked blir problemstillingen:

*Hvordan påvirker valg av prognosemodell prediksjonsnøyaktigheten for månedlig offhire for fartøy innenfor samme offshoresegment, når maskinlæringsmodeller sammenlignes med tradisjonelle tidsseriemodeller?*

For å presisere hva prognoseproblemet faktisk består av, deles problemstillingen videre inn i to delspørsmål:

1.  Hvilken modell predikerer størrelsen på neste måneds offhire mest presist?

2.  Hvordan varierer modellene i sine framtidsprognoser for månedlig offhire på korte og lengre prognosehorisonter?

Delspørsmålene retter oppmerksomheten mot både historisk prediksjonsnøyaktighet og modellbaserte fremtidsbilder. Studien vurderer dermed både hvilke modeller som treffer best mot kjente observasjoner, og hvordan de samme modellene gir ulike prognoser når de brukes fremover i tid.

## Avgrensninger

Studien avgrenses til 16 anonymiserte fartøy som opererer innenfor samme offshoresegment. Fartøy utenfor dette segmentet inngår ikke i analysen. Avgrensningen er valgt for å sikre størst mulig sammenlignbarhet i operasjonelle rammebetingelser og kontraktsforhold, selv om fartøyene ikke nødvendigvis tilhører én og samme fartøytype.

Analysen er videre avgrenset til prediksjon av månedlig offhire i prosent, vist som andel dager uten kontrakt i hver fartøy måned. Målet brukes som en praktisk indikator på offhire og redusert kontraktsutnyttelse. Det skilles ikke videre mellom ulike årsakskategorier innen offhire, fordi formålet er å evaluere modellenes prediktive ytelse og ikke å analysere årsakssammenhenger.

Studien bygger på historiske operasjonelle data innen en definert tidsperiode. Eksterne forhold som energipriser, geopolitisk forhold og markedsendringer modelleres ikke eksplisitt, men inngår bare i den grad de er indirekte reflektert i observasjonene.

Fokuset ligger på sammenligning av modelltyper med hensyn til prediksjonsnøyaktighet. Implementeringskostnader, organisatoriske endringer og teknologisk integrasjon inngår ikke i analysen. Modellutvalget er avgrenset til fire modeller: to tradisjonelle metoder, SARIMA og eksponentiell glatting, og to KI-baserte modeller, XGBoost og LSTM. Modellene representerer ulike metodiske tilnærminger til prognostisering, fra statistiske modeller med eksplisitt tidsseriestuktur til maskinlæringsmodeller som kan håndtere mer komplekse og ikke-lineære mønstre. Utvalget er valgt for å belyse hvordan ulike modellfamilier presterer under samme operasjonelle og datamessige rammebetingelser (Hyndman & Athanasopoulos, Forecasting: principles and practice, 2021; Chen & Guestrin, 2016; Hochreiter & Schmidhuber, 1997).

## Antakelser

### Definisjon av offhire og begrepsbruk

Begrepet offhire brukes i denne oppgaven fordi det ikke finnes et helt presist norsk enkeltord som dekker betydningen slik den brukes i datagrunnlaget. I maritim sammenheng kan offhire vise til perioder der et fartøy ikke genererer kontraktsmessig inntekt eller ikke er tilgjengelig for ordinær kontraktsdrift. I denne studien brukes begrepet i en avgrenset operasjonell betydning, der offhire måles som prosentandel dager per måned registrert uten kontrakt. Begrepet oversettes derfor ikke konsekvent til ett norsk ord, men omtales også som redusert kontraktsutnyttelse der dette gir bedre lesbarhet. Det antas at denne registreringen er konsistent gjennom hele datamaterialet. Antakelsen er nødvendig for at variasjon i datasettet skal kunne tolkes som uttrykk for reelle operasjonelle forhold, og ikke som følge av endrede registreringsrutiner. Eventuelle strukturelle brudd i rapporteringspraksis fanges derfor ikke eksplisitt opp i analysen.

### Historiske mønstre inneholder prediktiv informasjon

Det antas at historiske operasjonelle data inneholder mønstre som kan brukes til å predikere fremtidig månedlig offhire. Denne antakelsen ligger til grunn for både tidsseriemodeller og maskinlæringsmodeller. Analysen vurderer derfor modellenes evne til å utnytte eksisterende historisk struktur, men ikke deres evne til å forutsi strukturelle brudd utenfor datagrunnlaget.

### Uavhengighet mellom fartøy

Det antas at observasjoner kan behandles som tilnærmet uavhengige mellom fartøy. Denne antakelsen forenkler modelleringen og gjør det mulig å sammenligne prediktiv ytelse uten å eksplisitt modellere flåteinteraksjoner. Eventuelle systematiske sammenhenger mellom fartøy, for eksempel felles teknisk design eller kontraktsstruktur, inngår dermed ikke eksplisitt i modellene.

### Markedsusikkerhet reflekteres i historiske data

Det antas at markedsmessig usikkerhet, herunder svingninger i aktivitet og rater, indirekte er reflektert i de historiske dataene. Studien modellerer ikke slike forhold eksplisitt, men forutsetter at deler av effekten er synlig i observerte offhire mønstre. Modellene evalueres derfor under historisk observerte markedsforhold, men ikke mot hypotetiske ekstreme scenarier.

### Prediktiv fremfor kausal analyse

Det antas at formålet med analysen er prediksjon og ikke kausal forklaring. Modellenes prestasjon vurderes derfor ut fra prediktiv nøyaktighet og ikke ut fra om variablene representerer direkte årsakssammenhenger. Konsekvensen er at studien kan sammenligne modeller på likt grunnlag, men ikke trekke kausale konklusjoner om hvilke forhold som skaper offhire.

## Bruk av kunstig intelligens

I arbeidet med denne oppgaven er generative KI-verktøy brukt som støtteverktøy i skrive-, kode- og analyseprosessen. OpenAI ChatGPT og Google Gemini er brukt til idéutvikling, strukturering av tekst, gjennomgang av faglig sammenheng, språkvask og forslag til presisering av formuleringer. Codex i Visual Studio Code er brukt som støtte i kodearbeidet, blant annet til utvikling, feilsøking og videre bearbeiding av Python-kode for dataklargjøring, modellering, resultattabeller og rapportstruktur. KI-verktøyene er brukt som hjelpemidler, ikke som erstatning for selvstendig faglig arbeid. Modellvalg, metodeoppsett, tolkning av resultater, diskusjon og endelige formuleringer er gjennomgått, vurdert og kvalitetssikret. KI-verktøy er også brukt som støtte i arbeid med kodebaserte visualiseringer og figurer, men de endelige figurene bygger på datagrunnlaget og modellresultatene fra analysen. Alle resultater som presenteres i oppgaven er kontrollert opp mot datagrunnlaget og modellkjøringene som ligger til grunn for analysen.

# Litteratur

Litteraturen om prognostisering i logistikk og forsyningskjeder viser et tydelig skifte fra et nesten ensidig fokus på klassiske statistiske modeller til et bredere metoderepertoar der maskinlæring og dyp læring inngår som reelle alternativer. Douaioui et al. (2024) viser i en nyere oversikt at veksten i KI-baserte prognosestudier har vært særlig sterk de siste årene, og at forskningen i økende grad retter seg mot datasett med ikke-lineariteter, strukturelle brudd og høy kompleksitet. Dette betyr likevel ikke at klassiske modeller har mistet sin relevans. Litteraturen peker snarere mot at modellvalg må forstås som et spørsmål om problemstruktur og datagenererende prosesser, ikke som et teknologisk kappløp der den mest avanserte modellen automatisk er best.

Denne nyanseringen blir tydelig i forskning på prognoser under ustabile omgivelser. Fildes et al. (2022) viser at forecasting etter COVID-19 i mindre grad kan bygge på en enkel videreføring av stabile historiske mønstre, og i større grad må håndtere brudd, skift og episodiske sjokk. M5-konkurransen har gitt et viktig empirisk grunnlag for samme diskusjon. Makridakis et al. (2022) viser at store sammenligningsstudier kan avdekke betydelige forskjeller i prediksjonsnøyaktighet når datastrukturen er rik og krevende. Samtidig advarer Kolassa (2022) mot å lese slike konkurranser som et generelt bevis på at mer komplekse modeller alltid gir høyest praktisk verdi. Forskningen peker dermed mot en vurdering der prediktiv nøyaktighet, robusthet, tolkbarhet og praktisk anvendbarhet må ses i sammenheng.

Direkte sammenligninger mellom statistiske modeller og maskinlæringsmodeller understøtter samme poeng. Schmid et al. (2025) viser i en simuleringsstudie for data-drevet logistikk at maskinlæringsmetoder særlig kommer til sin rett når datastrukturen er preget av ikke-linearitet, heterogenitet og forstyrrelser, mens tradisjonelle tidsseriemodeller fortsatt kan være fullt konkurransedyktige i mer regelmessige settinger. Forskningen gir dermed ikke støtte til en enkel antakelse om at modellkompleksitet i seg selv er et kvalitetsmål. Det sentrale blir i stedet å sammenligne modellfamilier under samme evalueringsoppsett og på samme datasett.

Innen maritim forskning er anvendelsen av slike modeller økende, men tematikken er fortsatt relativt smal. Kalafatelis et al. (2025) viser at KI i maritim sektor i stor grad er brukt innen prediktivt vedlikehold, med fokus på komponentfeil, tilstandsmonitorering og teknisk tilgjengelighet. Chu et al. (2024) viser at XGBoost kan forbedre prediksjoner av fartøyets tid i havn, mens Kjeldsberg og Munim (2024) demonstrerer at AutoML og maskinlæringsmodeller kan brukes til å predikere PSV-fraktrater i et marked preget av flere samtidige og ikke-lineære drivere. Felles for disse studiene er at de dokumenterer økende bruk av datadrevne modeller i maritime beslutningsproblemer, men de retter seg hovedsakelig mot teknisk vedlikehold, havneoperasjoner eller markedsrater.

Det er derfor fortsatt et tydelig forskningsgap knyttet til prognostisering av operasjonell offhire på fartøynivå i offshoresegmentet. I stedet for å teste én enkelt modell undersøker oppgaven hvordan to klassiske tidsseriemodeller, SARIMA og eksponentiell glatting, og to KI-baserte modeller, XGBoost og LSTM, presterer når de sammenlignes på samme datastruktur og med samme historiske evalueringslogikk. Historisk prediksjonsnøyaktighet brukes dermed som hovedgrunnlag for modellvurderingen, mens framtidsprognosene tolkes i lys av denne historiske testen.

# Teori

## Prognostisering som beslutningsstøtte

Prognostisering er sentralt i logistikk og operasjonsstyring fordi beslutninger om ressursbruk, vedlikehold, kontrakter og kapasitet må tas før framtidige utfall er kjent. En prognose er derfor ikke bare et estimat av en framtidig verdi, men et beslutningsverktøy som reduserer usikkerhet under operasjonelle begrensninger. I denne studien er dette viktig fordi formålet er prediksjon av månedlig offhire i prosent og ikke kausal forklaring av hvorfor variasjonen oppstår. Modellenes verdi vurderes dermed ut fra hvor godt de kan omsette historiske mønstre til praktisk beslutningsstøtte for framtidige perioder (Carbonneau, Laframboise, & Vahidov, 2008; Fildes, Kolassa, & Ma, 2022).

## Tidsserier og sentrale komponenter

En tidsserie er en sekvens av observasjoner $y_{t}$ ordnet i tid, der rekkefølgen er analytisk meningsbærende. Observasjoner som ligger nær hverandre i tid vil ofte være statistisk avhengige, og denne avhengigheten er selve grunnlaget for prognostisering. I klassisk tidsserieanalyse dekomponeres serien gjerne i fire hovedkomponenter: nivå, trend, sesong og et irregulært restledd. Nivå beskriver seriens typiske størrelse, trend beskriver en mer langsiktig utvikling, sesong beskriver gjentakende mønstre med fast periode, og restleddet representerer variasjon som ikke fanges av den systematiske strukturen.

For data om offhire er denne dekomponeringen relevant fordi materialet kan inneholde både perioder med stabilt lavt nivå, episodiske topper og kalender mønstre knyttet til drift og kontraktsforhold. Samtidig er det viktig å skille mellom modellfamiliene i hvordan de bruker slik struktur. Klassiske tidsseriemodeller forsøker å spesifisere den direkte gjennom differensiering, glatting og autokorrelasjon, mens maskinlæringsmodeller og sekvensmodeller i større grad forsøker å lære mønstrene indirekte fra data. Like fullt er begrepene nivå, trend, sesong og støy et nyttig felles språk for å forstå prognoseproblemet på tvers av modelltyper (Gardner, 1985; Hyndman & Khandakar, Automatic time series forecasting: The forecast package for R, 2008).

## Klassiske tidsseriemodeller

Klassiske tidsseriemodeller bygger på at framtidige observasjoner som kan estimeres ved å modellere den interne dynamikken i seriens egen historikk. I denne oppgaven representeres denne tradisjonen av SARIMA og eksponentiell glatting. Felles for dem er at de i hovedsak er univariate og lar prognosen bestemmes av tidligere observasjoner, eventuelle differensieringer og et begrenset sett av parametere eller tilstandskomponenter. Dette gjør dem relativt transparente sammenlignet med mer komplekse maskinlæringsmodeller.

En sentral styrke ved klassiske modeller er at antakelsene kan formuleres eksplisitt. Dersom tidsserien etter transformasjoner og differensieringer kan behandles som tilnærmet stabil, kan modeller med få parametere gi presise og tolkbare prognoser. Samtidig er denne styrken også en begrensning: når dataserien er sterkt uregelmessig, svært nulltung eller påvirkes av flere samtidige og ikke-lineære forhold, blir det vanskeligere å beskrive hele prognoseproblemet gjennom én eksplisitt tidsseriedynamikk. Valget av klassiske modeller i denne studien er derfor ikke begrunnet i at de nødvendigvis er enklest modellene, men at de representerer etablerte og brukte referansemodeller innen prognostisering (Gardner, 1985; Hyndman, Koehler, Snyder, & Grose, 2002; Hyndman & Khandakar, Automatic time series forecasting: The forecast package for R, 2008).

## SARIMA

SARIMA, seasonal autoregressive integrated moving average, utvider ARIMA-rammeverket til serier med sesongmønster. En SARIMA(p,d,q)(P,D,Q)\_s-modell kombinerer autoregressive ledd (p, P), differensiering (d, D) og glidende gjennomsnittsledd (q, Q) på både ordinært og sesongmessig nivå, der s er sesonglengden.

**Standard matematisk form**

|  |  |
|----|---:|
|
``` math
\Phi(B^{12})\phi(B)(1 - B)^{d}(1 - B^{12})^{D}y_{t} = \Theta(B^{12})\theta(B)\varepsilon_{t}
``` | (3.1) |

**Forklaring av symbolene**

Her er $y_{t}$ observert offhire i prosent i måned $t$, og $B$ er backshift-operatoren, slik at $By_{t} = y_{t - 1}$. Videre er $\phi(B)$ og $\theta(B)$ henholdsvis ikke-sesong autoregressivt og glidende gjennomsnittspolynom av orden $p$ og $q$, mens $\Phi(B^{12})$ og $\Theta(B^{12})$ er sesongpolynomer av orden $P$ og $Q$. Parameterne $d$ og $D$ angir ordinær og sesongmessig differensiering, $\varepsilon_{t}$ er et tilfeldig feilledd, og sesonglengden er satt til 12 fordi dataseriene er månedlige.

Ligning (3.1) viser at SARIMA kan kombinere kortsiktig dynamikk og et mulig årlig sesongsignal i samme modell. At sesongleddet skrives med B^{12} betyr konkret at modellen kan hente informasjon fra samme måned året før, altså fra observasjonen y\_{t-12}.

Teoretisk er SARIMA mest relevant når historikken inneholder en tidsstruktur som kan beskrives gjennom autokorrelasjon og gjentakende sesongmønstre. Modellen er derfor sterk når nivå, trend og sesong kan identifiseres relativt klart, og når en viktig del av prognoseproblemet ligger i seriens egen dynamikk. Samtidig er modellen sårbar dersom serien er kort, svært nulltung eller dominert av uregelmessige sprang, fordi antakelsen om en tilnærmet stabil tidsserieprosess da blir vanskeligere å opprettholde (Hyndman & Khandakar, Automatic time series forecasting: The forecast package for R, 2008).

## Eksponentiell glatting

Eksponentiell glatting bygger på en annen modelllogikk enn SARIMA, men er fortsatt en sentral klassisk metode innen prognostisering. Grunntanken er at nyere observasjoner skal tilleggs større vekt enn eldre observasjoner, der vektene avtar eksponentielt bakover i tid. I den moderne ETS-modeller beskrives tidsserien gjennom uobserverte komponenter for nivå, trend og eventuelt sesong, som oppdateres fortløpende når nye observasjoner blir tilgjengelige (Gardner, 1985; Hyndman, Koehler, Snyder, & Grose, 2002).

**Standard matematisk form**

|  |  |
|----|---:|
|
``` math
\mathcal{l}_{t} = \alpha(y_{t} - s_{t - 12}) + (1 - \alpha)(\mathcal{l}_{t - 1} + b_{t - 1})
``` | (3.2) |

|  |  |
|----|---:|
|
``` math
b_{t} = \beta(\mathcal{l}_{t} - \mathcal{l}_{t - 1}) + (1 - \beta)b_{t - 1}
``` | (3.3) |

|                                                                  |       |
|------------------------------------------------------------------|------:|
|
 ``` math
 s_{t} = \gamma(y_{t} - \mathcal{l}_{t}) + (1 - \gamma)s_{t - 12}
 ```                                                               | (3.4) |

|                                                                    |       |
|--------------------------------------------------------------------|------:|
|
 ``` math
 {\widehat{y}}_{t + 1|t} = \mathcal{l}_{t} + b_{t} + s_{t + 1 - 12}
 ```                                                                 | (3.5) |

**Forklaring av symbolene**

Her er $y_{t}$ observert offhire i prosent i måned $t$, $\mathcal{l}_{t}$ nivåkomponenten, $b_{t}$ trendkomponenten og $s_{t}$ sesongkomponenten. Parametrene $\alpha$, $\beta$ og $\gamma$ er glattingsparametere mellom 0 og 1 som styrer hvor raskt nivå, trend og sesong reagerer på ny informasjon. Prognosen ${\widehat{y}}_{t + 1|t}$ uttrykker forventet verdi neste måned gitt informasjon tilgjengelig ved tid $t$.

Ligningene (3.2) til (3.5) viser hvordan nivå, trend, sesong og én-stegsprognose oppdateres rekursivt. Referansen til t-12 representerer samme måned året før og gjør det mulig å fange opp eventuelle årlige sesongmønstre i dataserien. I motsetning til SARIMA modellerer ikke ETS tidsserien gjennom en eksplisitt autokorrelasjonsstruktur. Prognosen bygger i stedet på den løpende estimerte tilstanden i serien. Dette gjør modellen relativt transparent og godt egnet som sammenligningsmodell i prognosestudier. I denne studien er eksponentiell glatting særlig relevant fordi modellfamilien representerer en mellomposisjon mellom SARIMA og de mer komplekse maskinlæringsmodellene. Dersom nyere observasjoner av offhire faktisk bærer mest relevant informasjon om den nære framtiden, kan modellen gi stabile prognoser med relativt få parametere. Samtidig kan modellen få utfordringer dersom datasettet er preget av store sprang, høy volatilitet eller betydelig heterogenitet mellom fartøyene, fordi slike forhold er vanskeligere å beskrive gjennom glatte nivå, trend og sesongkomponenter (Gardner, 1985; Hyndman, Koehler, Snyder, & Grose, 2002)

## Maskinlæring og dyp læring i prognostisering

Maskinlæring og dyp læring representerer en mer datadrevet tilnærming til prognostisering enn klassiske tidsseriemodeller. I stedet for å anta en bestemt stokastisk struktur for serien, forsøker modellene å lære en funksjonell sammenheng mellom input x_t og output y_t direkte fra data. Dette gjør dem særlig relevante når problemet preges av ikke-linearitet, interaksjoner mellom flere trekk eller heterogenitet som er vanskelig å beskrive med få eksplisitte parametere.

I prognosesammenheng er det nyttig å skille mellom feature-baserte modeller og sekvensbaserte modeller. Feature-baserte modeller, som XGBoost, krever at tidsproblemet omformes til et sett av eksplisitte inputvariabler, for eksempel laggede verdier, rullerende gjennomsnitt og kalenderindikatorer. Sekvensbaserte modeller, som LSTM, lar i større grad modellen lære tidsavhengigheter direkte fra ordnede sekvenser av observasjoner. Felles for disse modellene er at de tilbyr høy fleksibilitet, men også høyere krav til datamengde, datakvalitet og modelloppsett enn de klassiske tidsseriemodellene (Carbonneau, Laframboise, & Vahidov, 2008; Douaioui, Oucheikh, Benmoussa, & Mabrouki, 2024).

## XGBoost

XGBoost er en trebasert maskinlæringsmodell bygget på gradient boosting. Modellen konstruerer en additiv funksjon der prediksjonen skrives som summen av mange beslutningstrær (Chen & Guestrin, 2016).

**Standard matematisk form**

|  |  |
|----|---:|
|
``` math
{\widehat{y}}_{i} = \sum_{k = 1}^{K}f_{k}(x_{i}),\quad\quad f_{k}\mathcal{\in F}
``` | (3.6) |

|  |  |
|----|---:|
|
``` math
\mathcal{L(}\phi) = \sum_{i = 1}^{n}l(y_{i},{\widehat{y}}_{i}) + \sum_{k = 1}^{K}\Omega(f_{k})
``` | (3.7) |

|  |  |
|----|---:|
|
``` math
\Omega(f_{k}) = \gamma T_{k} + \frac{1}{2}\lambda\left\| w_{k} \right\|^{2}
``` | (3.8) |

**Forklaring av symbolene**

Her er $y_{i}$ observert offhire i prosent for observasjon $i$, ${\widehat{y}}_{i}$ modellens prediksjon, og $x_{i}$ feature-vektoren som beskriver observasjonen. Hvert $f_{k}$ representerer et beslutningstre, $K$ er antall trær, og $\mathcal{F}$ er rommet av mulige regresjonstrær. Tapsfunksjonen $l(y_{i},{\widehat{y}}_{i})$ måler prediksjonsfeil, mens regulariseringsleddet $\Omega(f_{k})$ straffer unødvendig komplekse trær gjennom antall terminale noder $T_{k}$ og bladvekter $w_{k}$, styrt av parametrene $\gamma$ og $\lambda$.

Ligningene (3.6) til (3.8) tydeliggjør at XGBoost ikke modellerer tidsserien gjennom én eksplisitt serieformel, men gjennom en additiv prediksjonsfunksjon med regularisering. Tidsavhengigheten må derfor legges inn via features som blant annet kan peke ett år tilbake gjennom for eksempel lag_12. Chen og Guestrin (2016) viser at XGBoost kombinerer høy prediksjonsstyrke med regularisering, effektiv trebygging og god håndtering av datasett med mange nullverdier. Regulariseringsleddet i objektfunksjonen gjør at modellen ikke bare forsøker å minimere treningsfeil, men også straffer unødig komplekse trær.

I denne studien er XGBoost en relevant kontrast til både SARIMA og eksponentiell glatting: dersom offhire best forstås som et ikke-lineært panelproblem med fartøyspesifikke effekter, bør modellen ha et teoretisk fortrinn. Dersom den relevante strukturen først og fremst ligger i den interne dynamikken i hver tidsserie, kan behovet for eksplisitt feature engineering bli en begrensning (Chen & Guestrin, 2016).

## LSTM

LSTM, long short-term memory, er en rekurrent sekvensmodell utviklet for å lære avhengigheter over lengre tidshorisonter. Hochreiter og Schmidhuber (1997) utviklet modellen for å håndtere problemet med at vanlige rekurrente nettverk har vansker med å bevare eller propagere relevant informasjon over lange sekvenser. Kjernen i LSTM er en minnecelle c_t og en skjult tilstand h_t, styrt av inngangs-, glemme- og utgangsporter.

**Standard matematisk form**

|                                                             |       |
|-------------------------------------------------------------|------:|
|
 ``` math
 f_{t} = \sigma(W_{f}\lbrack h_{t - 1},x_{t}\rbrack + b_{f})
 ```                                                          | (3.9) |

|                                                             |        |
|-------------------------------------------------------------|-------:|
|
 ``` math
 i_{t} = \sigma(W_{i}\lbrack h_{t - 1},x_{t}\rbrack + b_{i})
 ```                                                          | (3.10) |

|  |  |
|----|---:|
|
``` math
{\widetilde{c}}_{t} = tanh(W_{c}\lbrack h_{t - 1},x_{t}\rbrack + b_{c})
``` | (3.11) |

|                                                                 |        |
|-----------------------------------------------------------------|-------:|
|
 ``` math
 c_{t} = f_{t} \odot c_{t - 1} + i_{t} \odot {\widetilde{c}}_{t}
 ```                                                              | (3.12) |

|                                                             |        |
|-------------------------------------------------------------|-------:|
|
 ``` math
 o_{t} = \sigma(W_{o}\lbrack h_{t - 1},x_{t}\rbrack + b_{o})
 ```                                                          | (3.13) |

|                                 |        |
|---------------------------------|-------:|
|
 ``` math
 h_{t} = o_{t} \odot tanh(c_{t})
 ```                              | (3.14) |

|                                            |        |
|--------------------------------------------|-------:|
|
 ``` math
 {\widehat{y}}_{t + 1} = W_{y}h_{t} + b_{y}
 ```                                         | (3.15) |

**Forklaring av symbolene**

Her er $x_{t}$ inputvektoren ved tid $t$, $f_{t}$ forget gate, $i_{t}$ input gate, ${\widetilde{c}}_{t}$ kandidat for ny celletilstand, $c_{t}$ celletilstanden, $o_{t}$ output gate og $h_{t}$ den skjulte tilstanden. Symbolene $\sigma( \cdot )$ og $tanh( \cdot )$ betegner aktiveringsfunksjoner, $\odot$ er elementvis multiplikasjon, og $W_{\cdot}$ samt $b_{\cdot}$ er vekter og biasledd som estimeres i treningen. Prognosen ${\widehat{y}}_{t + 1}$ uttrykker forventet offhire i prosent i neste måned gitt sekvensinformasjonen fram til tid $t$.

Ligningene (3.9) til (3.15) viser portmekanismen som gjør at LSTM kan bevare, oppdatere og filtrere historisk informasjon over tid. I denne oppgaven skjer dette over et observasjonsvindu på 12 måneder, slik at modellen alltid ser ett helt år bakover før den predikerer neste måned.

Portene regulerer hvilken informasjon som beholdes, oppdateres og eksponeres videre i sekvensen. Dette gjør LSTM fundamentalt forskjellig fra feature-baserte maskinlæringsmodeller, fordi tidsavhengigheten læres direkte i nettverket i stedet for å måtte spesifiseres gjennom manuelle laggede variabler. I denne oppgaven er LSTM teoretisk relevant fordi modellen representerer den mest fleksible og sekvensorienterte måten å lære mønstre i offhiredataen på. Dersom fartøyenes historikk inneholder lange eller sammensatte avhengigheter som ikke lett lar seg beskrive gjennom eksplisitte lagg og lineære parametere, bør LSTM kunne fange dette. Samtidig kommer denne fleksibiliteten med klare kostnader i form av større datakrav, høyere treningssensitivitet og lavere tolkbarhet enn både klassiske modeller og XGBoost. Modellen er derfor interessant nettopp fordi den utfordrer spørsmålet om hvor mye kompleksitet datasettet faktisk bærer (Hochreiter & Schmidhuber, 1997).

## Modellvalg og sammenligningskriterier

Et sentralt spørsmål i prognoseteori er hvordan modeller skal sammenlignes på en metodisk rettferdig måte. En modell kan ikke vurderes isolert ut fra hvor avansert den er, men må vurderes ut fra hvor godt den kan predikerer på data som ikke er brukt i modelltilpasningen. Makridakis et al. (2022) viser gjennom M5-konkurransen at modellvalg kan ha stor betydning for prognosenøyaktighet, men også at rangeringen avhenger av datastruktur, evalueringsoppsett og valgte feilmål. Kolassa (2022) understreker samtidig at modellkompleksitet ikke i seg selv er et kvalitetsmål. En mer kompleks modell er derfor ikke nødvendigvis bedre dersom datagrunnlaget er begrenset, uregelmessig eller ikke inneholder nok stabil struktur til at kompleksiteten kan utnyttes. En modell som treffer godt i rolige perioder kan likevel bomme mye når offhire øker brått, mens en modell som reagerer sterkt på tidligere topper kan overvurdere nivået i mer normale måneder. Modellene bør derfor sammenlignes på samme historiske testperiode, med samme informasjonsgrunnlag og samme prognosehorisont. Dette gjør at forskjeller i resultater i større grad kan knyttes til modellstruktur og ikke til ulikt evalueringsoppsett.

Hyndman og Athanasopoulos (2021) viser at valg av feilmål er en sentral del av prognoseevaluering, fordi ulike mål kan gi ulike perspektiver på modellprestasjon. I denne oppgaven vurderes modellene ved hjelp av MAE, RMSE, sMAPE og MASE. Flere evalueringsmål brukes fordi de belyser ulike sider av prognosekvalitet. MAE, mean absolute error, måler gjennomsnittlig absolutt avvik mellom faktisk og predikert verdi. Fordelen med MAE er at det er lett å tolke fordi feilen uttrykkes i samme enhet som variabelen som predikeres. MAE gir dermed et direkte mål på hvor mye modellen i gjennomsnitt bommer, uten at positive og negative feil utligner hverandre (Hyndman & Athanasopoulos, Forecasting: principles and practice, 2021). RMSE, root mean squared error, måler kvadratroten av gjennomsnittlig kvadrert feil. Siden feilene kvadreres før gjennomsnittet beregnes, gir RMSE større vekt til store avvik enn MAE. RMSE brukes derfor ofte når store prediksjonsfeil vurderes som særlig uønskede. Samtidig betyr dette at RMSE kan bli sterkt påvirket av få ekstreme observasjoner, og bør derfor ofte tolkes sammen med andre feilmål (Hyndman & Athanasopoulos, Forecasting: principles and practice, 2021). sMAPE, symmetric mean absolute percentage error, uttrykker prognosefeil som et prosentbasert mål relativt til størrelsen på faktisk og predikert verdi. Fordelen med prosentbaserte mål er at de kan gjøre det lettere å sammenligne feil på tvers av serier med ulik skala. Samtidig kan slike mål være krevende å tolke når verdiene er svært lave eller nær null. sMAPE bør derfor brukes som et supplement, og ikke nødvendigvis som eneste grunnlag for modellrangering (Hyndman & Athanasopoulos, Forecasting: principles and practice, 2021). MASE, mean absolute scaled error, er et skalert feilmål som sammenligner modellens absolutte feil med feilen fra en enkel naiv referansemodell. Hyndman og Koehler (2006) introduserer MASE som et alternativ til prosentbaserte feilmål, blant annet fordi det kan brukes på tvers av serier med ulik skala. En MASE-verdi under 1 betyr at modellen i gjennomsnitt gjør det bedre enn den naive referansen, mens en verdi over 1 betyr at modellen gjør det dårligere enn referansen.

Samlet gir disse evalueringsmålene et bredere vurderingsgrunnlag enn én enkelt metrikk alene. Modellvalget vurderes dermed ikke bare ut fra lavest mulig feil i én metrikk, men ut fra en samlet vurdering av prediksjonsnøyaktighet, robusthet, tolkbarhet og datakrav. Dette er i tråd med Schmid et al. (2025), som viser at sammenligningen mellom statistiske modeller og maskinlæringsmodeller må forstås i lys av problemstruktur og datagrunnlag, heller enn som et generelt spørsmål om hvilken modellfamilie som alltid er best.

# Casebeskrivelse

Denne studien tar utgangspunkt i Simon Møkster Shipping AS, et rederi som opererer innenfor offshoresegmentet. I denne konteksten er fartøyene en sentral del av verdikjeden fordi de understøtter aktivitetene på norsk sokkel gjennom kontrakter og leveranser. For et rederi i dette segmentet er høy operasjonell tilgjengelighet avgjørende, siden fartøyenes verdi i stor grad er knyttet til evnen til å opprettholde kontrakter og levere stabile tjenester over tid.

Et sentralt problem i denne sammenhengen er offhire. I denne oppgaven måles offhire som prosent dager per måned registrert uten kontrakt, og dette brukes som en indikator på redusert kontraktsutnyttelse. Målet kan påvirkes av tekniske feil, vedlikehold, sertifikatforhold, operasjonelle avvik eller perioder uten kontrakt. For Simon Møkster Shipping AS innebærer slike perioder ikke bare direkte økonomiske konsekvenser, men også økt usikkerhet i planlegging, ressursutnyttelse, vedlikeholdsvurderinger og kontraktsoppfølging. Mer presise prognoser kan derfor ha praktisk verdi som beslutningsstøtte.

Studien er avgrenset til 16 fartøy innenfor samme segment i rederiet. Dette er gjort for å sikre at analysen bygger på observasjoner fra fartøy med relativt like operasjonelle rammebetingelser, selv om fartøyene ikke nødvendigvis er av samme type. Et viktig trekk ved caset er samtidig at offhire sjeldent skyldes én enkelt faktor. Teknisk tilstand, driftsmønster, kontraktssituasjon og markedsforhold kan virke sammen, noe som gjør caset relevant for å sammenligne både tradisjonelle tidsseriemodeller og KI-baserte prognosemodeller.

Samtidig opererer rederiet i et marked preget av betydelig volatilitet. Offshoreaktiviteten påvirkes ikke bare av tekniske og operative forhold, men også av svingninger i energipriser, investeringsnivå, kontraktsaktivitet og globale markedsforhold. Dette gjør planlegging av fartøyutnyttelse og tilgjengelighet mer krevende, og øker den praktiske verdien av prognoser for offhire. Studien modellerer ikke slike eksterne forhold eksplisitt, men de utgjør en viktig del av konteksten som gjør prognoseproblemet relevant (Menon Economics, 2026).

Figur 1 viser samlet offhire per måned aggregert på tvers av alle fartøy. Figuren gir et første bilde av hvor stabilt eller ujevnt materialet faktisk er over tid.


*Figur 1. Samlet offhire per måned fra april 2021 til mars 2026, målt som summen av offhire i prosentpoeng på tvers av alle fartøy.*

Den samlede tidsserien viser tydelige topper og rolige perioder, heller enn en jevn utvikling. Den høyeste enkeltmåneden i datasettet er januar 2022 med 283 summerte prosentpoeng, mens det også finnes to måneder der samlet offhire er null. Dette viser at caset ikke dreier seg om en stabil serie, men om et fenomen som opptrer episodisk og ujevnt.

Mens figur 1 viser totalnivået i materialet, viser figur 2 hvordan variasjonen fordeler seg mellom fartøyene og over tid.


*Figur 2. Heatmap som viser offhire per fartøy og måned. Mørkere farger indikerer høyere offhire, mens lyse felt indikerer lave eller null registreringer.*

Heatmapet viser at offhire i liten grad er jevnt fordelt mellom fartøyene. Enkelte fartøy har gjentatte og tydelige topper over flere perioder, mens andre i hovedsak har nullregistreringer. Særlig skiller Fartøy 10, Fartøy 8 og Fartøy 2 seg ut med flere markerte utslag. Figuren synliggjør også at 2021 er et oppstartsår fra april og at 2026 foreløpig bare dekker årets tre første måneder. Samlet viser dette et konkret operasjonelt beslutningsproblem der prognostisering kan være verdifullt, men også metodisk krevende.

# Metode og data

## Metode

Studien bygger på kvantitativ og casebasert forskningsdesign der historiske offhire data analyseres for å undersøke hvordan ulike prognosemodeller påvirker prediksjonsnøyaktigheten for fartøy innenfor samme offshoresegment. Simon Møkster Shipping AS brukes som casebedrift, men fartøyene er anonymisert i analysen og omtales derfor som Fartøy 1 til Fartøy 16. Formålet med metoden er ikke å identifisere kausale sammenhenger, men å sammenligne hvor godt ulike modelltyper kan predikere fremtidig månedlig offhire i prosent basert på historiske observasjoner.

Studien bygger på kvantitative sekundærdata, der analyseenheten er ett fartøy i én bestemt måned. Forskningsdesignet er valgt fordi samme datastruktur kan brukes til å evaluere både klassiske tidsseriemodeller og KI-baserte modeller under samme evalueringsoppsett. Arbeidet består av dataklargjøring, deskriptiv analyse av datasettet, eksplisitt train/test-splitt, modellering, historisk validering og fremtidsprognoser.

### Modellutvalg og evalueringsoppsett

Valget av modeller bygger på at studien skal sammenligne to klassiske og to KI-baserte modeller under samme betingelser. SARIMA og eksponentiell glatting representerer mer strukturerte modeller som i hovedsak henter prognoseinformasjon fra seriens egen historikk, nivå, sesong og tidsavhengighet. XGBoost og LSTM representerer mer fleksible tilnærminger som kan håndtere ikke-linearitet, heterogenitet og mer komplekse mønstre, men som samtidig stiller høyere krav til feature-konstruksjon, treningsoppsett og datamengde. Sammenligningen gjør det mulig å undersøke om datastrukturen i caset best beskrives gjennom eksplisitt tidsseriedynamikk eller gjennom mer fleksible datadrevne modeller.

For å sikre en rettferdig sammenligning estimeres og evalueres alle modellene på samme historiske tidsvindu og med samme ekspanderende én-stegs prognoselogikk. Ingen av modellene får dermed tilgang til mer fremtidsinformasjon enn de andre. Forskjeller i resultatene kan derfor i større grad knyttes til modellstruktur enn til ulikt testdesign. MAE brukes som hovedmål fordi metrikken er lett å tolke i samme skala som målvariabelen og mindre dominert av enkeltmåneder med svært store feil enn RMSE. RMSE brukes som støttemål fordi den tydeliggjør hvor hardt modellene straffes for store bommerter, mens sMAPE brukes som et prosentbasert supplement. Siden datasettet er nulltungt og inneholder flere nær-nullverdier, tolkes sMAPE med forsiktighet og brukes ikke som eneste grunnlag for modellrangering. MASE brukes som et ekstra skalert støttemål, der feilene sammenlignes med en naiv lag-1-prognose beregnet fra treningshistorikken. Hyperparametere og modelloppsett for XGBoost og LSTM ble holdt relativt moderate og faste gjennom hele sammenligningen. Målet er derfor en sammenlignbar modellstudie, ikke en full optimaliseringsstudie av hver enkelt modellfamilie.

På tvers av modellene ble en 12-månedersmekanisme inkludert fordi datasettet består av månedlige observasjoner, og ett år er derfor den mest naturlige kandidaten for eventuell sesongvariasjon. I SARIMA og eksponentiell glatting betyr dette at modeller med periode 12 kan vurderes, mens det i XGBoost kommer inn gjennom lag_12 og rolling_mean_12, og i LSTM gjennom et sekvensvindu på 12 måneder. Valget innebærer ikke at sterk og stabil sesong ble antatt på forhånd, men at modellene skulle få mulighet til å utnytte et årlig mønster dersom det faktisk fantes i dataene.

De to delspørsmålene behandles gjennom samme overordnede regresjonsoppsett, der modellene predikerer månedlig offhire i prosent. Det første delspørsmålet besvares gjennom historisk én-stegs modelltesting mot kjente observasjoner i testperioden. Dette gjør det mulig å vurdere hvilken modell som predikerer neste måneds offhire mest presist. Det andre delspørsmålet besvares gjennom fremtidsprognoser på 1, 3, 6 og 12 måneders horisont. Disse prognosene kan ikke evalueres mot faktiske observasjoner, men brukes til å sammenligne hvordan modellene gir ulike estimater for mulig fremtidig utvikling.

Datasettet renses først og omstruktureres til long-format før det deles i et treningssett for perioden 2021-04 til 2024-12 og et testsett for perioden 2025-01 til 2026-03. Deretter estimeres fire modeller, SARIMA, Eksponentiell glatting, XGBoost og LSTM, på historiske data og evalueres mot usette observasjoner i testperioden. Alle modellene evalueres med samme ekspanderende én-stegs prognoselogikk og vurderes ved hjelp av MAE, RMSE, sMAPE og MASE. Etter den historiske testfasen brukes hele datasettet som grunnlag for fremtidsprognoser med horisonter på 1, 3, 6 og 12 måneder.

Denne metoden er valgt fordi den gir et transparent og sammenlignbart grunnlag for å vurdere modellvalg på samme problem og samme datagrunnlag. Ved å holde testperioden utenfor den historiske evalueringsfasen blir det mulig å vurdere modellenes generaliseringsevne, ikke bare deres tilpasning til treningsdataene. Samtidig har opplegget klare begrensninger. Datamaterialet består av sekundærdata som ikke kan verifiseres fullt ut eksternt, dataserien er relativt kort, og materialet er preget av mange nullperioder og betydelig variasjon mellom fartøyene. Funnene bør derfor primært forstås som case-spesifikke for dette segmentet.

## Data

### Datagrunnlag

Datagrunnlaget i denne studien består av kvantitative sekundærdata mottatt som et anonymisert uttrekk fra Simon Møkster Shipping AS. Datasettet er lagret som en CSV-fil og inneholder månedlige registreringer av offhire uttrykt som prosentandel dager uten kontrakt for 16 fartøy som opererer innenfor samme segment i offshorenæringen. I oppgaven brukes dette målet som en indikator på redusert kontraktsutnyttelse. Siden datasettet er anonymisert, omtales fartøyene i oppgaven som Fartøy 1 til Fartøy 16.

Tidsperioden i datasettet strekker seg fra april 2021 til mars 2026. Materialet er organisert i seks årsblokker, én for hvert år fra 2021 til 2026, og dekker totalt 16 fartøy. Råfilen består av 125 rader inkludert årsrader, kolonneoverskrifter og tomme skillerader. Etter rensing og omstrukturering til long-format, der hver rad representerer ett fartøy i én bestemt måned, består analysegrunnlaget av 902 observasjoner. Fordelingen over tid er 135 observasjoner i 2021, 180 observasjoner per år i 2022, 2023, 2024 og 2025, samt 47 observasjoner i 2026. At 2021 og 2026 har færre observasjoner skyldes at dataserien starter i april 2021 og foreløpig bare går til og med mars 2026.

Databehandlingen har bestått av flere trinn. Først ble tomme rader, overskriftsrader og verdier markert som N/A fjernet fra analysegrunnlaget. Deretter ble prosentverdier standardisert til numeriske verdier, blant annet ved å omforme komma til punktum i desimaltall. Til slutt ble datasettet gjort om fra et bredt årsformat til et analyseklar long-format med variablene fartøy, måned, dato, offhire-verdi og eventuelle spesielle behov eller krav. Denne omstruktureringen var nødvendig for å kunne bruke både tradisjonelle tidsseriemodeller og maskinlæringsmodeller på samme datagrunnlag.

For selve modelleringen ble datasettet i tillegg delt i et eksplisitt trenings- og testsett. Treningsdelen dekker perioden fra april 2021 til desember 2024, mens testdelen dekker januar 2025 til mars 2026. Denne tidsbaserte splitten er valgt for å sikre at modellene evalueres på observasjoner som ligger etter treningsperioden i tid, og dermed ikke får tilgang til informasjon fra framtiden under evalueringen. I det rensede analysegrunnlaget gir dette 675 observasjoner i treningssettet og 227 observasjoner i testsettet. Av disse 227 testobservasjonene tilhører to Fartøy 16, som først kommer inn helt mot slutten av dataserien og derfor ikke har tilstrekkelig treningshistorikk til å inngå i en rettferdig sammenligning på tvers av alle modellene. Testsettet bygger derfor på 225 observasjoner fordelt på 15 fartøy.

Rådataene fra virksomheten er ikke offentlig tilgjengelige. Analysefilene som brukes i oppgaven er anonymisert og dokumentert i prosjektets GitHub-repo, se vedleggstabell A2. Dette skyldes at materialet bygger på interne virksomhetsdata. For å sikre transparens beskrives derfor variablene, tidsperioden, databehandlingen og antall observasjoner eksplisitt i oppgaven, slik at analyseopplegget kan forstås og etterprøves metodisk selv om rådataene ikke publiseres åpent.

Tabell 1 oppsummerer hovedtrekkene i datagrunnlaget. Tabellen tydeliggjør at dataserien ikke dekker hele 2021 og 2026, noe som må tas hensyn til når nivå og variasjon i materialet senere tolkes.

| Tabell 1. Datadekning | Verdi |
|----|----|
| Tidsperiode | 2021-04 til 2026-03 |
| Antall kalendermåneder | 60 |
| Antall fartøy | 16 |
| Antall rå rader i CSV | 125 |
| Antall årsblokker | 6 |
| Antall rensede observasjoner | 902 |
| Observasjoner i 2021 | 135 |
| Observasjoner i 2022 | 180 |
| Observasjoner i 2023 | 180 |
| Observasjoner i 2024 | 180 |
| Observasjoner i 2025 | 180 |
| Observasjoner i 2026 | 47 |
| Kommentar | 2021 starter i april, og 2026 går foreløpig bare til mars |


### Deskriptiv analyse av datasettet

De overordnede figurene i casebeskrivelsen viser at offhire varierer både over tid og mellom fartøy. I denne delen utdypes derfor datastrukturen mer detaljert for å vise hvilke trekk ved materialet som er særlig relevante før modellering. Formålet er ikke å evaluere prognosemodeller, men å beskrive datagrunnlaget på en måte som gjør det tydelig hvorfor ulike modellfamilier kan forventes å prestere forskjellig.

#### Variasjon mellom fartøy

Figur 3 rangerer fartøyene etter gjennomsnittlig månedlig offhire i hele observasjonsperioden. Fordi 2021 og 2026 er ufullstendige år, er gjennomsnittlig månedlig nivå et mer informativt mål enn rene totalsummer.


*Figur 3. Gjennomsnittlig månedlig offhire per fartøy for hele perioden april 2021 til mars 2026.*

Figuren viser at offhire i stor grad er konsentrert rundt et mindre antall fartøy. Fartøy 10 har høyest gjennomsnittlig offhire med 16,86 prosent, tett fulgt av Fartøy 8 med 15,53 prosent og Fartøy 2 med 15,43 prosent. Samtidig viser figuren at flere fartøy ligger svært lavt gjennom hele perioden, og at Fartøy 13 og Fartøy 16 ikke har registrert offhire i datagrunnlaget. Det skal nevnes at fartøy 16 ikke har like mange observasjoner, siden dette fartøyet kommer inn i rederiet i februar 2026.

Tabell 2 oppsummerer de fem fartøyene med høyest gjennomsnittlig offhire og viser samtidig hvor stor andel av månedene som likevel er nullmåneder. Tabellen viser at selv fartøyene med høyest gjennomsnittlig nivå har mange måneder uten registrert offhire.

| Tabell 2. Fartøy med høyest gjennomsnittlig nedetid | Gjennomsnittlig offhire (%) | Maks offhire (%) | Andel nullmåneder (%) |
|----|---:|---:|---:|
| Fartøy 10 | 16.86 | 100.00 | 56.67 |
| Fartøy 8 | 15.53 | 93.00 | 58.33 |
| Fartøy 2 | 15.43 | 100.00 | 68.33 |
| Fartøy 9 | 7.34 | 100.00 | 88.33 |
| Fartøy 5 | 6.97 | 88.00 | 76.67 |

Figur 4 utdyper denne variasjonen ved å vise fordelingen av offhire for hvert fartøy gjennom hele perioden, ikke bare gjennomsnittsnivået.


*Figur 4. Boksplott som viser median, kvartiler og ekstreme observasjoner for offhire per fartøy.*

Boksplottet viser at datasettet er tydelig nulltungt og høyreskjevt. For de fleste fartøy ligger medianen på eller svært nær null, mens noen få måneder trekker nivået kraftig opp. Dette betyr at gjennomsnitt alene ikke gir et fullstendig bilde av datastrukturen. Offhire fremstår i stedet som et fenomen med mange nullperioder kombinert med enkelte markerte topper, noe som er viktig å ta hensyn til i senere modellvalg.

#### Tidsutvikling for fartøy med høyest offhire

For å undersøke om fartøyene med høyest gjennomsnittlig offhire følger like eller ulike mønstre over tid, viser figur 5 de fem fartøyene med høyest gjennomsnittsnivå.


*Figur 5. Historiske tidsserier vist som fem separate delpaneler for fartøyene med høyest gjennomsnittlig månedlig offhire i datasettet.*

Figuren viser at selv de mest aktive fartøyene ikke følger et jevnt eller stabilt mønster. Offhire opptrer i perioder med konsentrerte topper, avbrutt av lange intervaller med lave eller null registreringer. Fartøy 10 og Fartøy 8 har flere kraftige utslag gjennom perioden, mens Fartøy 5 i større grad preges av sporadiske topper adskilt av lange perioder med null. Samlet peker dette dermed mot at datasettet er preget av både sterk heterogenitet mellom fartøy og betydelig tidsmessig uregelmessighet.

# Modellering

I denne delen bygges, testes og evalueres modellene kun på historiske data. Fremtidsprognosene behandles i en egen seksjon og inngår derfor ikke i modellbeskrivelsene nedenfor. For å sikre en sammenlignbar evaluering testes alle modellene på samme historiske periode og med samme ekspanderende én-stegs prognoselogikk måned for måned.

Det felles evalueringsoppsettet er oppsummert i tabell 3. Resultatene må derfor forstås som en sammenligning av modellprestasjon for den delen av datasettet der alle modellene kan evalueres under samme betingelser. Hoved sammenligningen bygger på 15 fartøy og totalt 225 fartøy-måneder i testsettet.

| Tabell 3. Felles evalueringsoppsett | Verdi                                 |
|-------------------------------------|---------------------------------------|
| Målvariabel                         | Månedlig nedetid i prosent per fartøy |
| Treningsperiode                     | 2021-04 til 2024-12                   |
| Testperiode                         | 2025-01 til 2026-03                   |
| Prognosehorisont i test             | 1 måned per steg                      |
| Evalueringslogikk                   | Ekspanderende 1-stegs prognose        |
| Sammenligningsnivå                  | Fartøynivå                            |
| Hovedmål                            | MAE                                   |
| Støttemål                           | RMSE, sMAPE og MASE                   |
| Datagrunnlag i hovedtest            | 15 fartøy og 225 prediksjoner         |


## SARIMA

I denne studien brukes SARIMA fartøyvis, slik at hver tidsserie modelleres som en egen månedlig serie for offhire. Modellen skal fange opp treghet i fartøyets historiske utvikling, eventuelle sesongmønstre over året og kortsiktige avvik som ikke kan forklares av nivå alene. Den er derfor særlig egnet når neste måneds offhire kan forstås som avhengig av både tidligere måneder og gjentakende sesongstruktur. Konkret modelleres den månedlige offhireserien gjennom autoregressive ledd, differensiering, glidende gjennomsnitt og eventuelle sesongkomponenter med periode 12.

SARIMA bygger på den etablerte Box-Jenkins-tradisjonen for tidsseriemodellering, men anvendes her på fartøynivå fremfor på aggregert flåtenivå (Ljung & Box, 1978). For hvert fartøy ble det først kontrollert at tidsserien hadde tilstrekkelig historikk og variasjon. Deretter ble ADF brukt som støtte for differensieringsvalg, før et begrenset parameterrom for SARIMA modellen ble estimert og rangert med AIC, BIC og parsimoni. Sesongledd med periode 12 ble inkludert der datastrukturen tydet på at det var relevant, men ikke tvunget frem kun fordi dataene er månedlige. Når et slikt ledd er med, peker det eksplisitt tilbake på observasjonen y\_{t-12}, altså samme måned året før. Dersom observasjonen fra samme måned året før inneholder lite informasjon eller ofte er null, vil sesongbidraget også bli begrenset (Ljung & Box, 1978).

Fartøy 13 hadde en konstant nullserie i treningsperioden og ble derfor håndtert som en eksplisitt konstant-baseline innen samme evalueringsramme. De øvrige 14 fartøyene fikk estimerte SARIMA-modeller. For et representativt fartøy med høy historisk offhire, Fartøy 2, viser tabell 4 de beste kandidatmodellene. Den valgte modellen for dette fartøyet ble SARIMA (2,0,0)(1,0,0,12). Fartøy 2 er brukt som representativt eksempel i figurer og modelltabeller, men modellene er ikke trent utelukkende på dette fartøyet. For SARIMA og eksponentiell glatting estimeres egne modeller for hvert fartøy i datasettet, mens XGBoost og LSTM trenes på det samlede fartøy-måned-panelet. Fartøy 2 brukes derfor kun som et illustrativt eksempel i presentasjonen av modellene.

| Tabell 4. Beste kandidatmodeller for representativt fartøy (Fartøy 2) | AIC | BIC |
|----|---:|---:|
| SARIMA(2,0,0)(1,0,0,12) | 291.43 | 297.16 |
| SARIMA(2,0,1)(1,0,0,12) | 293.40 | 300.57 |
| SARIMA(2,0,2)(1,0,0,12) | 295.17 | 303.77 |
| SARIMA(1,0,0)(1,0,0,12) | 297.89 | 302.29 |
| SARIMA(1,0,1)(1,0,0,12) | 299.85 | 305.71 |

Figur 6 og 7 viser ACF og PACF for det representative fartøyet etter valgt transformasjon. Figur 8 viser residualene for samme eksempel. På tvers av de estimerte SARIMA-modellene var Ljung-Box-p-verdiene over 0.05 (Ljung & Box, 1978), noe som taler for at det ikke gjenstår tydelig autokorrelasjon i residualene.


*Figur 6. ACF for representativt fartøy (Fartøy 2) brukt som støtte i modellidentifikasjonen.*


*Figur 7. PACF for representativt fartøy (Fartøy 2) brukt som støtte i modellidentifikasjonen.*


*Figur 8. Residualdiagnostikk for valgt SARIMA-modell på Fartøy 2. Figuren viser både residualforløp og residualfordeling.*

Figur 9 viser hvordan den valgte modellen treffer i testperioden for det representative fartøyet. Figuren brukes ikke som hovedgrunnlag for modellvurderingen, men som en visuell kontroll på hvordan modellen følger utviklingen i testperioden.


*Figur 9. Historiske testprediksjoner for SARIMA på Fartøy 2. Grå linje viser treningsdata, blå linje faktisk testforløp og oransje linje modellens prediksjoner.*

## Eksponentiell glatting

I denne oppgaven estimeres eksponentiell glatting fartøyvis på månedlig offhire i prosent. Nivå, trend og sesong oppdateres fortløpende når nye månedsobservasjoner blir tilgjengelige, slik at nyere observasjoner får større vekt enn eldre observasjoner. Dette gjør modellen relevant som en relativt konservativ, men samtidig responsiv benchmark i et datasett der flere serier har lange nullperioder avbrutt av mer uregelmessige utslag. Modellen beskriver den månedlige offhire gjennom separate komponenter for nivå, trend og sesong, uten å kreve samme grad av eksplisitt modellstruktur som SARIMA.

Eksponentiell glatting fungerte i denne studien som den mest konservative klassiske modellen. Også denne modellen ble estimert per fartøy. I stedet for å tvinge én spesifikasjon på alle serier ble et lite og bevisst begrenset sett av additive ETS-varianter vurdert: nivåmodell (ANN), nivå med trend (AAN) og nivå med trend og sesong (AAA). For konstante serier ble det brukt en eksplisitt konstant baseline. At bare ett fartøy endte med en eksplisitt AAA-modell, tyder på at sterke og stabile sesongmønstre med periode 12 har begrenset empirisk støtte i store deler av datasettet.

Tabell 5 oppsummerer hvilke spesifikasjoner som faktisk ble valgt. Resultatet viser at datasettet i liten grad støtter mer komplekse glattemodeller: 13 fartøy endte med ANN, 1 fartøy med AAA, og 1 fartøy med konstant-baseline. Det ble ikke valgt noen AAN-modeller i analysen.

| Tabell 5. Valgt ETS-spesifikasjon i analysen | Antall fartøy |
|----------------------------------------------|--------------:|
| ANN                                          |            13 |
| AAA                                          |             1 |
| CONST                                        |             1 |

Residualdiagnostikken viser at ETS fungerer rimelig godt for mange fartøy, men svakere enn SARIMA på enkelte serier. Særlig Fartøy 2 og Fartøy 7 fikk Ljung-Box-p-verdier under 0.05, noe som indikerer at restautokorrelasjon ikke var like godt håndtert i alle tilfeller (Ljung & Box, 1978). Figur 10 viser testforløpet for det representative fartøyet.


*Figur 10. Historiske testprediksjoner for eksponentiell glatting på Fartøy 2.*

## XGBoost

I denne studien brukes XGBoost som en feature-basert panelmodell på fartøy-måned-nivå, snarere enn som en klassisk univariat tidsseriemodell. Modellens inputvariabler består av laggede observasjoner, rullerende gjennomsnitt, rullerende standardavvik, kalenderkomponenter og fartøyspesifikke trekk. Modellen predikerer dermed neste måneds offhire ved å lære mønstre i konstruerte tidsserie-features, heller enn å spesifisere tidsavhengigheten eksplisitt i én serieformel. Prognoseproblemet behandles som en supervisert regresjonsoppgave der prediksjonen uttrykkes som summen av flere beslutningstrær.

XGBoost ble satt opp som én global modell på fartøy-måned-paneldata. Modellen fikk et eksplisitt feature-set som bare brukte informasjon tilgjengelig før hver testmåned. Dermed følger også denne modellen samme ekspanderende én-stegs prognoselogikk som de klassiske modellene.

Feature-settet er oppsummert i tabell 6. Poenget var å gi modellen både kortsiktig historikk, glattet historikk og kalenderinformasjon, samtidig som fartøyspesifikke forskjeller kunne fanges gjennom vessel og special_flag. Modellen kan derfor ikke forstås som et enkelt gjennomsnitt av tidligere nullmåneder. XGBoost bruker samtidig korte lags, lag_12, rullerende gjennomsnitt, rullerende standardavvik og fartøyspesifikke indikatorer. I et nulltungt datasett kan amplituden likevel bli dempet, fordi flere av feature-ene får moderate verdier når historikken domineres av lange nullperioder og få ekstreme topper.

| Tabell 6. XGBoost-featuregrupper | Innhold |
|----|----|
| Historiske lags | lag_1, lag_2, lag_3, lag_6, lag_12 |
| Rullerende nivå | rolling_mean_3, rolling_mean_6, rolling_mean_12 |
| Rullerende variasjon | rolling_std_3, rolling_std_6, rolling_std_12 |
| Kalender | month_num, quarter_num, year_num, time_idx, month_sin, month_cos |
| Kategoriske trekk | vessel, special_flag |

Hyperparametrene ble holdt faste gjennom hele testoppsettet, som vist i tabell 7. Dette er et bevisst valg for å prioritere sammenlignbarhet mellom modellfamiliene fremfor maksimal optimalisering av XGBoost isolert sett.

| Tabell 7. XGBoost-hyperparametre | Verdi |
|----------------------------------|------:|
| n_estimators                     |   200 |
| max_depth                        |     4 |
| learning_rate                    |  0.05 |
| subsample                        |  0.90 |
| colsample_bytree                 |  0.90 |

Figur 11 viser at modellen i hovedsak bygger på kort og mellomlang historikk. lag_1 er viktigst, men også rolling_mean_12, rolling_mean_6, rolling_mean_3 og enkelte fartøyindikatorer bidrar mye, mens lag_12 bidrar mer moderat. Dette er konsistent med at problemet både har tidsseriepreg og tydelig fartøyheterogenitet, men også med at samme måned året før ikke alltid gir et sterkt signal i dette materialet.


*Figur 11. Viktigste features i referansemodellen for XGBoost estimert på treningsperioden. Laggede verdier og rullerende gjennomsnitt dominerer.*

Figur 12 viser den historiske testytelsen for det representative fartøyet. Sammenlignet med de klassiske modellene framstår XGBoost som mer fleksibel, men fortsatt sårbar i perioder med svært uregelmessige topper.


*Figur 12. Historiske testprediksjoner for XGBoost på Fartøy 2. Figuren viser modellens evne til å følge nivåendringer uten eksplisitt tidsseriemodell.*

## LSTM

I denne studien brukes LSTM som en global sekvensmodell på fartøy-måned-data, der et observasjonsvindu på 12 måneder brukes til å predikere neste måned. Inputvariablene inneholder historisk offhire samt kalender- og fartøyrelatert informasjon. Den skjulte tilstanden representerer kortsiktig informasjon fra sekvensen, mens celletilstanden fungerer som modellens mer langvarige hukommelse. LSTM er derfor relevant fordi modellen kan lære tidsavhengigheter over flere måneder uten at disse må spesifiseres manuelt gjennom laggede variabler. I det konkrete modelloppsettet ble hver observasjon representert som en sekvens på 12 måneder, med fire inputfeatures per tidssteg: den historiske målvariabelen, som i kodegrunnlaget er kalt offhire_days, men som i analysefilen representerer månedlig offhire-prosent, month_sin, month_cos og special_flag. All skalering ble estimert på treningsdata. Modellen ble deretter re-trent måned for måned i samme ekspanderende testoppsett som de øvrige modellene. Også her betyr 12-månedersvinduet at modellen får se ett helt års historikk, men ikke at samme måned året før automatisk gis størst vekt. Hvis store deler av sekvensen består av nuller eller rolige perioder, kan resultatet bli et mer dempet prognosenivå.

Det konkrete oppsettet er vist i tabell 8.

| Tabell 8. LSTM-oppsett i analysen | Verdi |
|----|----|
| Sekvenslengde | 12 måneder |
| Inputformat | samples x timesteps x features |
| Inputfeatures | offhire_days, month_sin, month_cos, special_flag |
| LSTM-enheter | 32 |
| Dense-enheter | 16 |
| Batch size | 8 |
| Maks antall epoker | 100 |
| Tidlig stopping | Ja, med gjenoppretting av beste vekter |

Figur 13 viser treningshistorikken fra referansekjøringen på treningsperioden. Valideringstapet flater tidlig ut og begynner deretter å stige, noe som understøtter at tidlig stopping er nødvendig for å unngå overtilpasning.


*Figur 13. Trenings- og valideringstap for LSTM estimert på treningsperioden. Figuren viser at modellen lærer raskt, men at valideringstapet ikke forbedres videre etter de første epokene.*

Figur 14 viser testforløpet for det representative fartøyet. Som for XGBoost er modellen fleksibel, men den store fordelen over de beste klassiske modellene er ikke tydelig i dette datasettet.


*Figur 14. Historiske testprediksjoner for LSTM på Fartøy 2. Figuren viser at modellen følger nivåendringer relativt godt, men ikke tydelig bedre enn de sterkeste alternativene.*

Samlet viser modelleringen at alle fire modellfamiliene er bygget og testet innenfor samme historiske evalueringsramme. Forskjellen mellom modellene ligger derfor ikke i testdesignet, men i hvordan de håndterer samme datastruktur under samme betingelser.

## Oppsett for fremtidsprognoser

Etter at modellene var evaluert på den historiske testperioden, ble fremtidsprognosene gjennomført som en egen fase. I denne fasen ble hele datasettet til og med 2026-03 brukt som treningsgrunnlag, og første prognosemåned ble dermed 2026-04. For alle fire modellene ble det generert prognoser med horisonter på 1, 3, 6 og 12 måneder.

For de klassiske modellene ble prognosene generert fartøyvis som fler-stegsprognoser direkte fra den estimerte modellen. For XGBoost og LSTM ble prognosene generert iterativt måned for måned, slik at predikert verdi fra ett steg inngår som historisk input i neste prognosesteg. Dette gjør at alle modellene kan sammenlignes på samme fremtidige datovindu, samtidig som de beholder sin opprinnelige modellstruktur.

Den iterative logikken er viktig for hvordan nivå og utslag i fremtidsprognosene tolkes. Når modellene bruker sine egne tidligere prediksjoner som input, kan prognosebanene enten dempes eller forsterkes avhengig av signalene i de foregående stegene. I et nulltungt datasett vil dette ofte trekke modellene mot moderate nivåer, men for enkelte fartøy kan samme mekanisme også forsterke positive prognoser dersom rekursive features begynner å bygge på hverandre.

Fremtidsprognosene evalueres ikke med MAE, RMSE, sMAPE eller MASE, fordi faktiske observasjoner ikke finnes ennå. I stedet brukes de som modellbaserte indikasjoner på mulig fremtidig utvikling. Prognosene er punktprognoser uten usikkerhetsintervaller, og de bør derfor tolkes med økende forsiktighet jo lengre horisonten blir. For å gjøre resultatene sporbare ble det lagret egne prognosefiler både samlet og per horisont, samt figurer som summerer forventet offhire per måned og modell.

# Resultater

Resultatdelen er delt i to. Først presenteres resultatene fra den historiske modelltesten, som viser hvordan modellene presterer på historiske holdout-data. Deretter presenteres fremtidsprognosene for 1, 3, 6 og 12 måneder frem i tid. Denne todelingen er viktig fordi historisk test kan evalueres med feilmetrikker, mens fremtidsprognoser bare kan tolkes som modellbaserte estimater.

## Resultater fra historisk modelltesting

Alle modeller er evaluert på de samme 225 fartøy-månedene i testperioden fra januar 2025 til mars 2026. MAE brukes som hovedmål, mens RMSE, sMAPE og MASE brukes som støttemål. MASE er skalert mot en naiv lag-1-referanse beregnet fra treningshistorikken per fartøy. Siden datasettet også er svært nulltungt, må sMAPE tolkes med forsiktighet; metrikken kan bli ustabil eller svært høy når faktiske og predikerte verdier ligger nær null.

Tabell 9 viser det samlede testresultatet. SARIMA oppnår lavest MAE, lavest RMSE og lavest MASE i den historiske testen. XGBoost og LSTM ligger fortsatt svært nær hverandre på MAE og RMSE, mens eksponentiell glatting er svakest på disse to målene. MASE nyanserer likevel rangeringen bak SARIMA ved at eksponentiell glatting kommer bedre ut enn både XGBoost og LSTM når MASE evalueres.

| Tabell 9. Samlet testresultat for modellene | Antall prediksjoner | MAE | RMSE | sMAPE | MASE |
|----|---:|---:|---:|---:|---:|
| ARIMA/SARIMA | 225 | 6.15 | 16.78 | 100.44 | 0.91 |
| XGBoost | 225 | 7.35 | 17.40 | 182.98 | 1.20 |
| LSTM | 225 | 7.57 | 17.41 | 178.77 | 1.32 |
| Eksponentiell glatting | 225 | 8.37 | 17.95 | 168.62 | 1.18 |

MASE støtter dermed hovedfunnet om at SARIMA er den mest treffsikre modellen samlet sett, men viser også at bildet bak SARIMA er mer sammensatt enn MAE alene antyder. En MASE under 1 for SARIMA betyr at modellen i gjennomsnitt slår den naive lag-1-referansen. De øvrige modellene ligger over 1 og forbedrer dermed ikke denne enkle referansen i samme grad i testperioden.

Figur 15 visualiserer de samme MAE-resultatene som en samlet sammenligning. Figuren tydeliggjør at forskjellen mellom de tre beste modellene er relativt liten, men at SARIMA likevel kommer best ut i den historiske testen.


*Figur 15. Samlet MAE for de fire modellene i testperioden. Lavere verdi indikerer bedre prediksjonsnøyaktighet.*

Figur 16 viser hvordan MAE varierer mellom testmånedene. Ingen modell dominerer alle måneder fullstendig, men SARIMA er gjennomgående sterk og særlig stabil i flere av månedene med mer moderate nivåer. Samtidig viser figuren at alle modellene får høyere feil i måneder der offhire preges av store hopp eller episodiske utslag.


*Figur 16. MAE per måned i testperioden for de fire modellene. Figuren viser hvordan modellytelsen varierer over tid, ikke bare samlet.*

Figur 17 viser MAE per fartøy og modell som heatmap. Figuren tydeliggjør at de største feilene er konsentrert rundt noen få fartøy, særlig Fartøy 10, Fartøy 9 og Fartøy 8, mens flere fartøy med lav eller null offhire er enklere å predikere for alle modellene. Dette betyr at samlet modellrangering i stor grad påvirkes av hvor godt modellene håndterer de mest krevende fartøyene.


*Figur 17. Heatmap som viser MAE per fartøy og modell i testperioden. Mørkere felt indikerer høyere prediksjonsfeil.*

Resultatene fra den historiske modelltesten peker mot tre hovedobservasjoner. For det første har SARIMA best samlet testytelse i dette datasettet. For det andre presterer XGBoost og LSTM konkurransedyktig, men uten å gi en tydelig gevinst over den beste klassiske modellen. For det tredje varierer feilene mer mellom fartøyene enn mellom modeller, noe som viser at samlet modellrangering i stor grad påvirkes av de mest krevende fartøyene.

## Resultater fra fremtidsprognoser

Etter den historiske testfasen ble alle fire modellene estimert på hele datasettet til og med 2026-03. Prognosevinduet dekker dermed perioden 2026-04 til 2027-03. For å holde hovedteksten lesbar presenteres tabellene nedenfor som samlet prognostisert offhire per måned på tvers av de 15 fartøyene som inngår i prognosegrunnlaget. De detaljerte fartøyvise prognosene er dokumentert i filregisteret i vedleggstabell A2, blant annet gjennom future_predictions.csv og future_predictions_12m_pivot.csv. I hovedteksten presenteres derfor bare summerte prognoser per måned og modell.

### Prognose 1 måned frem

Tabell 10 viser prognosen for april 2026, altså én måned frem i tid. Allerede på denne korte horisonten gir modellene ulike nivåestimater. XGBoost gir høyest samlet prognose med 102.19 prosentpoeng, mens eksponentiell glatting ligger lavest med 53.37. På fartøynivå gir tre av fire modeller høyest prognose for Fartøy 10, mens SARIMA har den høyeste enkeltprognosen på Fartøy 9 med 42.93.

| Tabell 10. Samlet prognostisert offhire 1 måned fram | Eksponentiell glatting | LSTM | SARIMA | XGBoost |
|----|---:|---:|---:|---:|
| 2026-04 | 53.37 | 91.60 | 76.35 | 102.19 |


*Figur 18. Samlet prognostisert offhire i april 2026 for de fire modellene.*

### Prognose 3 måneder frem

Tabell 11 viser at forskjellene øker raskt når horisonten forlenges til tre måneder. Eksponentiell glatting ligger nærmest flatt gjennom hele vinduet, mens LSTM beveger seg moderat nedover. SARIMA og særlig XGBoost estimerer langt høyere nivåer i mai og juni. Ved utgangen av juni 2026 er forskjellen mellom høyeste og laveste modell over 180 prosentpoeng. Resultatene viser dermed at modellene utvikler ulike prognosebaner allerede på tre måneders horisont.

| Tabell 11. Samlet prognostisert offhire 3 måneder fram | Eksponentiell glatting | LSTM | SARIMA | XGBoost |
|----|---:|---:|---:|---:|
| 2026-04 | 53.37 | 91.60 | 76.35 | 102.19 |
| 2026-05 | 53.69 | 77.58 | 138.96 | 181.85 |
| 2026-06 | 54.00 | 66.39 | 154.27 | 238.57 |


*Figur 19. Samlet prognostisert offhire fra april til juni 2026 for de fire modellene.*

### Prognose 6 måneder frem

Tabell 12 viser prognosene fra april til september 2026, altså seks måneder frem i tid. Også her fremstår eksponentiell glatting som den mest konservative modellen, med et nesten uendret totalnivå fra måned til måned. LSTM faller tydelig utover sommeren, mens SARIMA varierer mer og beholder flere markerte topper. XGBoost ligger gjennomgående høyest og holder seg over 150 i alle måneder unntatt april. Resultatet viser at ETS og delvis LSTM gir mer dempede prognosebaner, mens XGBoost gir høyere nivåer gjennom store deler av seksmånedersvinduet.

| Tabell 12. Samlet prognostisert offhire 6 måneder fram | Eksponentiell glatting | LSTM | SARIMA | XGBoost |
|----|---:|---:|---:|---:|
| 2026-04 | 53.37 | 91.60 | 76.35 | 102.19 |
| 2026-05 | 53.69 | 77.58 | 138.96 | 181.85 |
| 2026-06 | 54.00 | 66.39 | 154.27 | 238.57 |
| 2026-07 | 54.32 | 50.80 | 113.85 | 191.26 |
| 2026-08 | 54.63 | 41.29 | 125.77 | 155.66 |
| 2026-09 | 54.95 | 25.67 | 58.83 | 206.98 |


*Figur 20. Samlet prognostisert offhire fra april til september 2026 for de fire modellene.*


### Prognose 12 måneder frem

Tabell 13 viser det fulle tolvmånedersvinduet frem til mars 2027. Her blir modellforskjellene svært tydelige. Eksponentiell glatting holder seg nesten helt flatt mellom 53.37 og 56.84, mens LSTM først faller og deretter stiger moderat igjen mot slutten av perioden. SARIMA beholder et mer bølgende og sesongpreget forløp med tydelige topper i mai-juni 2026 og januar-februar 2027. XGBoost skiller seg klart ut med de høyeste prognosenivåene, med en topp på 547.73 prosentpoeng i februar 2027. Dette viser også at forklaringen på modellatferden ikke kan reduseres til at alle modellene glatter mot null: noen blir flate, noen blir dempet, og noen kan bli klart forsterkende over lengre horisonter.

| Tabell 13. Samlet prognostisert offhire 12 måneder fram | Eksponentiell glatting | LSTM | SARIMA | XGBoost |
|----|---:|---:|---:|---:|
| 2026-04 | 53.37 | 91.60 | 76.35 | 102.19 |
| 2026-05 | 53.69 | 77.58 | 138.96 | 181.85 |
| 2026-06 | 54.00 | 66.39 | 154.27 | 238.57 |
| 2026-07 | 54.32 | 50.80 | 113.85 | 191.26 |
| 2026-08 | 54.63 | 41.29 | 125.77 | 155.66 |
| 2026-09 | 54.95 | 25.67 | 58.83 | 206.98 |
| 2026-10 | 55.26 | 24.81 | 16.84 | 310.44 |
| 2026-11 | 55.58 | 39.80 | 27.08 | 363.72 |
| 2026-12 | 55.89 | 54.01 | 14.65 | 416.52 |
| 2027-01 | 56.21 | 66.39 | 121.88 | 522.01 |
| 2027-02 | 56.52 | 67.61 | 149.42 | 547.73 |
| 2027-03 | 56.84 | 64.86 | 127.60 | 353.12 |


*Figur 21. Samlet prognostisert offhire fra april 2026 til mars 2027 for de fire modellene.*

Samlet viser fremtidsprognosene at modellene gir ulike fremtidsbilder, særlig på lengre horisonter. På kort sikt peker modellene mot at offhire fortsatt vil være konsentrert rundt noen få fartøy, men på lengre sikt varierer både nivå og utviklingsform betydelig. De historiske testresultatene blir derfor en viktig tolkningsramme: prognosene bør ikke leses isolert, men i lys av hvilke modeller som faktisk presterte best på historiske holdout-data. Dette er også viktig for skillet mellom spørsmålet om når neste offhire kommer og hvor stor den blir. Modellene gir først og fremst punktprognoser for nivået i neste fartøy-måned, mens tidsdimensjonen må tolkes indirekte gjennom om prognosen er null eller positiv.

# Diskusjon

Diskusjonsdelen tolker funnene opp mot problemstillingen, tidligere forskning og studiens metodiske forutsetninger. Formålet er ikke å gjenta resultatene, men å vurdere hva de faktisk betyr for modellvalg, praktisk beslutningsstøtte og i hvilken grad funnene kan generaliseres utover den konkrete casen.

## Modellvalg og prediksjonsnøyaktighet

Det tydeligste hovedfunnet i studien er at valg av prognosemodell har betydning for prediksjonsnøyaktigheten, men at økt modellkompleksitet ikke automatisk gir bedre resultater. I den historiske testen oppnådde SARIMA lavest MAE, RMSE og MASE, og fremstår derfor som den sterkeste modellen i dette konkrete evalueringsoppsettet. XGBoost og LSTM var konkurransedyktige på MAE, men ga ikke en tydelig forbedring sammenlignet med SARIMA. Eksponentiell glatting fungerte som en nyttig referansemodell, men kom samlet svakere ut på MAE og RMSE, selv om modellen presterte noe bedre enn XGBoost og LSTM på MASE.

Dette funnet er interessant fordi det nyanserer en vanlig forventning om at KI-baserte modeller nødvendigvis vil prestere best når datagrunnlaget er komplekst. Resultatene viser i stedet at det ikke er modellens kompleksitet alene som avgjør prediksjonsnøyaktigheten, men hvor godt modellstrukturen passer til datagrunnlaget og prognoseproblemet. Dette samsvarer med Schmid et al. (2025), som viser at forskjeller mellom statistiske modeller og maskinlæringsmodeller må forstås i lys av datastruktur, problemtype og evalueringsoppsett. Funnene støtter også Kolassa (2022), som argumenterer for at høyere modellkompleksitet ikke automatisk gir høyere praktisk verdi.

En mulig forklaring på at SARIMA kom best ut, er at modellen estimeres fartøyvis og dermed kan tilpasses den interne tidsstrukturen i hver enkelt serie. I et datasett der offhire varierer betydelig mellom fartøy, kan en fartøyvis modell ha en fordel fordi den ikke nødvendigvis må lære ett felles mønster på tvers av hele panelet. Dette ser særlig ut til å ha betydning for de mest utslagsgivende fartøyene i testperioden. På fartøynivå var SARIMA tydelig sterkere enn de andre modellene på flere av seriene som bidro mest til samlet feilnivå, blant annet Fartøy 9, Fartøy 10 og Fartøy 2. Dermed skyldes ikke SARIMA sitt samlede fortrinn bare små forskjeller på tvers av alle fartøy, men også at modellen håndterte enkelte av de mest krevende seriene bedre enn alternativene.

Samtidig bør ikke resultatene tolkes som at SARIMA dominerer alle deler av datasettet. Modellrangeringen varierer mellom fartøy, og både XGBoost og LSTM viser konkurransedyktige resultater på enkelte serier. XGBoost presterer best på noen fartøy og er marginalt bedre enn SARIMA på Fartøy 8, mens LSTM er relativt konkurransedyktig på flere fartøy med lavere eller mer stabil offhire. Dette viser at de KI-baserte modellene faktisk fanger opp deler av strukturen i datagrunnlaget, selv om de ikke gir en samlet forbedring i den historiske testen. Resultatet bør derfor ikke tolkes som en generell avvisning av maskinlæring i offhire-prognostisering, men som et uttrykk for at disse modellene ikke ga tilstrekkelig merverdi under de konkrete datamessige og metodiske betingelsene i denne oppgaven.

Et sentralt poeng er derfor at kompleksiteten i datasettet ikke nødvendigvis er av en type som KI-modellene fullt ut kan utnytte. Selv om datamaterialet er ujevnt, nulltungt og preget av episodiske topper, betyr ikke dette automatisk at mer fleksible modeller får et bedre læringsgrunnlag. Tvert imot kan korte tidsserier, mange nullobservasjoner og få gjentakende mønstre gjøre det vanskelig for modeller som XGBoost og LSTM å skille mellom reelle prediktive signaler og tilfeldig variasjon. Dette er særlig relevant for LSTM, som teoretisk kan lære sekvensielle avhengigheter over tid, men som samtidig stiller høyere krav til datamengde og stabilitet i mønstrene som skal læres (Hochreiter & Schmidhuber, 1997). XGBoost har på sin side høy fleksibilitet gjennom trebasert gradient boosting, men er avhengig av at de konstruerte feature-ene faktisk inneholder tilstrekkelig prediktiv informasjon (Chen & Guestrin, 2016).

Samlet sett viser resultatene at problemstillingen ikke kan besvares med at én modellfamilie generelt er best. Valg av modell påvirker prediksjonsnøyaktigheten, men effekten av modellvalget avhenger av forholdet mellom modellstruktur og datagrunnlag. I denne casen fremstår SARIMA som det mest forsvarlige førstevalget fordi modellen ga lavest historisk prediksjonsfeil og samtidig er relativt transparent sammenlignet med de KI-baserte alternativene. Samtidig viser de konkurransedyktige resultatene til XGBoost og LSTM at KI-baserte modeller fortsatt kan være relevante, særlig dersom fremtidige analyser får tilgang til lengre tidsserier, flere forklaringsvariabler eller rikere informasjon om kontrakter, tekniske forhold og markedssituasjon.

## Datastruktur, markedskontekst og modellprestasjon

En sentral forklaring på resultatene ligger i selve datamaterialet. Den deskriptive analysen viste at datasettet er nulltungt, høyreskjevt og preget av store forskjeller mellom fartøyene. Offhire opptrer ofte som episodiske topper snarere enn som jevne mønstre over tid. Dette betyr at modellene ikke bare konkurrerer om å lære et stabilt nivå, en trend eller et sesongmønster, men om å håndtere et prognoseproblem der lange perioder med nullverdier avbrytes av enkelte kraftige utslag. Når forskjellene mellom fartøyene samtidig er store, blir modellprestasjon også et spørsmål om heterogenitet mellom fartøy, og ikke bare om tidsavhengighet i én samlet serie.

Denne datastrukturen bidrar til å forklare hvorfor SARIMA kom best ut i den historiske testen. SARIMA ble estimert fartøyvis, og modellen kunne dermed tilpasses den interne tidsstrukturen i hver enkelt serie. I et datasett med store forskjeller mellom fartøyene kan dette være en fordel, fordi modellen ikke trenger å lære ett felles mønster for hele panelet. Samtidig viste residualdiagnostikken for SARIMA ingen tydelig restautokorrelasjon etter valgt grense for de estimerte fartøyseriene, noe som tyder på at modellene i rimelig grad fanget opp den tidsavhengigheten som faktisk var til stede i treningsdataene. At SARIMA samtidig oppnådde lavest MAE, RMSE og MASE i testperioden, styrker tolkningen av at en relativt strukturert fartøyvis modell passet godt til dette datagrunnlaget.

Resultatene betyr likevel ikke at datamaterialet var enkelt for klassiske tidsseriemodeller. Offhire-seriene er korte, ujevne og preget av mange nullobservasjoner, noe som gjør det krevende å identifisere stabile sesong- eller trendmønstre. Dette er særlig relevant for eksponentiell glatting. I analysen ble de fleste fartøyene modellert med en enkel ANN-spesifikasjon, mens bare ett fartøy fikk en eksplisitt modell med både trend og sesong. Det tyder på at sterk og stabil sesongstruktur ikke hadde bred empirisk støtte i materialet. Eksponentiell glatting fremstår derfor som en konservativ modell i denne studien: den gir stabile og relativt dempede prediksjoner, men har begrenset evne til å fange opp brå episodiske topper.

Den samme datastrukturen kan også forklare hvorfor XGBoost og LSTM ikke fikk en tydelig fordel, til tross for at de er mer fleksible modeller. Maskinlærings- og dyp læringsmodeller kan ha et teoretisk fortrinn når datagrunnlaget inneholder ikke-lineære mønstre, interaksjoner og heterogenitet som er vanskelige å beskrive med klassiske tidsseriemodeller (Carbonneau, Laframboise, & Vahidov, 2008; Douaioui, Oucheikh, Benmoussa, & Mabrouki, 2024; Schmid, Roidl, Kirchheim, & Pauly, 2025). I denne studien ser det likevel ut til at fleksibiliteten ikke ble fullt utnyttet. En mulig forklaring er at kompleksiteten i datasettet i stor grad består av korte serier, mange nullmåneder og få gjentakende topper. Slike forhold gir ikke nødvendigvis et rikt læringsgrunnlag for mer komplekse modeller. For XGBoost avhenger prestasjonen av at laggede verdier, rullerende gjennomsnitt, variasjonsmål og fartøyindikatorer inneholder stabil prediktiv informasjon. For LSTM avhenger prestasjonen av at sekvensene inneholder mønstre som kan læres over tid. Når store deler av historikken består av nullverdier, og toppene opptrer uregelmessig, blir begge deler metodisk krevende.

Den felles 12-månedersstrukturen er også viktig. Siden dataene er månedlige, er ett år den mest naturlige perioden å undersøke for mulig sesongvariasjon. Samtidig viser resultatene at en 12-månedersstruktur ikke automatisk innebærer at det finnes et sterkt sesongmønster. Dersom observasjonen fra samme måned året før ofte er null, eller dersom tidligere topper ikke gjentar seg på samme tidspunkt året etter, vil både sesongledd, lag_12 og 12-måneders sekvensvinduer kunne gi begrenset informasjon. Dette bidrar til å forklare hvorfor flere modeller gir lave eller moderate prognosenivåer i deler av forløpet, selv om modelloppsettet i utgangspunktet er metodisk rimelig.

Resultatene må i tillegg forstås i lys av markedskonteksten. Offshoresegmentet er preget av svingninger i aktivitet, investeringsnivå, rater og kontraktsforhold, og slike forhold kan påvirke fartøyenes kontraktsutnyttelse og risiko for offhire (Menon Economics, 2026). I denne studien modelleres ikke slike eksterne drivere eksplisitt. De inngår bare indirekte i den grad de allerede er reflektert i historiske offhire-observasjoner. Dette er en viktig begrensning, fordi brå markedsendringer, nye kontrakter, tekniske hendelser eller endrede operasjonelle prioriteringer kan påvirke fremtidig offhire uten at mønsteret nødvendigvis finnes tydelig i historikken.

Samlet viser dette at modellprestasjonene ikke kan forstås isolert fra datagrunnlaget. SARIMA sitt fortrinn i den historiske testen skyldes trolig ikke at klassiske modeller generelt er bedre enn KI-baserte modeller, men at en fartøyvis tidsseriemodell passet godt til et kort, heterogent og nulltungt datamateriale. XGBoost og LSTM kan fortsatt være relevante i videre arbeid, men resultatene tyder på at slike modeller først vil få et tydeligere potensial dersom datagrunnlaget utvides med lengre tidsserier, flere observasjoner og eksterne forklaringsvariabler som kontraktsstatus, tekniske indikatorer og markedsforhold.

## Fremtidsprognoser og praktisk tolkning

Fremtidsprognosene må tolkes annerledes enn resultatene fra den historiske modelltesten. I testperioden finnes faktiske observasjoner, og modellene kan derfor vurderes direkte ved hjelp av prediksjonsfeil. For prognoseperioden finnes det derimot ingen kjent fasit, og resultatene må derfor forstås som modellbaserte fremoverskrivninger snarere enn verifiserte utfall. Dette skillet er sentralt, fordi en modell som gir høye eller tydelige fremtidsprognoser ikke nødvendigvis er mer treffsikker enn en modell som gir mer moderate prognoser. Fremtidsprognosene bør derfor tolkes i lys av den historiske testytelsen, der SARIMA samlet sett hadde lavest MAE, RMSE og MASE.

Det mest sentrale funnet i fremtidsprognosene er at modellene gir ulike fremtidsbilder, og at forskjellene øker når prognosehorisonten forlenges. På kort horisont varierer modellene først og fremst i nivå, mens de på lengre horisonter også utvikler ulike prognosebaner. Eksponentiell glatting gir den mest stabile og konservative prognosen, LSTM beveger seg mot et mer dempet nivå, SARIMA gir et mer varierende forløp, mens XGBoost gir de høyeste prognosene på lengre horisont. Dette viser at modellvalg ikke bare påvirker historisk prediksjonsnøyaktighet, men også hvordan mulig fremtidig offhire fremstilles når modellene brukes fremover i tid.

Særlig XGBoost-prognosen på 12 måneders horisont bør tolkes med forsiktighet. I februar 2027 gir XGBoost en samlet prognose på 547,73 prosentpoeng, mens SARIMA ligger på 149,42, LSTM på 67,61 og eksponentiell glatting på 56,52. Denne forskjellen er så stor at XGBoost-prognosen ikke bør tolkes som et direkte uttrykk for at modellen med høy sikkerhet forventer et ekstremt offhire-nivå. Den bør heller forstås som et mulig utslag av hvordan en feature-basert maskinlæringsmodell kan oppføre seg i rekursive fler-stegsprognoser. Når modellen predikerer flere måneder frem i tid, brukes tidligere predikerte verdier videre som input i senere prognosesteg. For XGBoost betyr dette at laggede verdier, rullerende gjennomsnitt, variasjonsmål og fartøyspesifikke effekter gradvis kan bygge på modellens egne tidligere prediksjoner. Dersom slike prediksjoner først beveger seg oppover for enkelte fartøy, kan denne effekten forsterkes over tid og trekke den samlede prognosen kraftig opp.

Dette betyr ikke nødvendigvis at XGBoost-prognosen er ubrukelig, men at den må forstås som et uttrykk for høy modellusikkerhet på lang horisont. Prognosen kan fungere som et varsel om at modellen identifiserer mønstre som, gitt dens egen logikk, kan gi økende offhire. Samtidig bør den ikke brukes som selvstendig beslutningsgrunnlag uten operasjonell vurdering. Når en modell gir et så markant avvik fra de øvrige modellene, er selve modelluenigheten et viktig funn. Den viser at fremtidsestimatene er følsomme for modellstruktur, og at usikkerheten øker betydelig når prognosene strekkes langt frem i tid.

Amplituden i prognosene må derfor tolkes modellspesifikt. Eksponentiell glatting gir et nesten flatt forløp fordi modellen i hovedsak viderefører et glattet nivå fra historikken. Dette er metodisk konsistent med at de fleste fartøyseriene ikke viste sterk og stabil trend eller sesong. LSTM gir et mer dempet forløp, trolig fordi 12-månederssekvensene i stor grad inneholder nullverdier og moderate nivåer, noe som trekker prognosene nedover. SARIMA gir mer variasjon fordi modellen estimeres fartøyvis og kan videreføre enkelte tidsserie- og sesongmønstre der disse finnes. XGBoost gir derimot et mer forsterkende langtidsforløp, noe som trolig henger sammen med modellens bruk av laggede og rullerende features i en rekursiv prognosestruktur.

Sett opp mot det andre delspørsmålet viser fremtidsprognosene at modellene varierer betydelig i sine estimater for månedlig offhire på korte og lengre prognosehorisonter. På 1 og 3 måneders horisont kan prognosene ha praktisk verdi som støtte for kortsiktig kapasitetsplanlegging og oppfølging av fartøy med høy historisk offhire. På 6 og særlig 12 måneders horisont bør prognosene i større grad forstås som usikre modellbaner eller scenario indikasjoner. Dette er særlig viktig fordi studien ikke inkluderer eksterne forklaringsvariabler som kontraktsendringer, tekniske hendelser, energipriser eller markedsrater. Slike forhold kan påvirke fremtidig offhire, men fanges bare opp i modellene dersom de allerede er indirekte reflektert i historiske observasjoner.

For Simon Møkster Shipping AS betyr dette at fremtidsprognosene først og fremst bør brukes som beslutningsstøtte, ikke som automatiske beslutningsgrunnlag. De korte prognosehorisontene kan bidra til å identifisere fartøy og perioder der operasjonell oppfølging bør prioriteres. De lengre prognosehorisontene bør derimot brukes mer forsiktig, særlig når modellene spriker kraftig. En praktisk tolkning er at stor avstand mellom modellene ikke nødvendigvis viser hvilket fremtidig nivå som er mest sannsynlig, men at usikkerheten rundt fremtidig offhire er høy. Dermed kan modelluenighet i seg selv brukes som et signal om behov for nærmere faglig og operasjonell vurdering.

## Metodiske styrker og svakheter

Studien har flere metodiske styrker. For det første sammenlignes alle modellene på samme historiske tidsvindu og med samme ekspanderende én-stegs evalueringslogikk. Dette styrker den interne sammenlignbarheten, fordi modellene vurderes mot de samme observasjonene og under samme informasjonsbetingelser. Forskjeller i prediksjonsnøyaktighet kan dermed i større grad knyttes til modellstruktur enn til ulikt testdesign. Dette er viktig der evalueringsoppsett og valg av testperiode kan ha stor betydning for modellrangeringen (Hyndman & Athanasopoulos, Forecasting: principles and practice, 2021; Makridakis, Spiliotis, & Assimakopoulos, 2022).

For det andre kombinerer oppgaven to klassiske tidsseriemodeller og to KI-baserte modeller. Dette gir et bredere sammenligningsgrunnlag enn om bare én modellfamilie var vurdert. Modellutvalget gjør det mulig å undersøke om månedlig offhire best håndteres gjennom eksplisitt tidsseriedynamikk, slik som i SARIMA og eksponentiell glatting, eller gjennom mer fleksible datadrevne metoder, slik som XGBoost og LSTM. En slik sammenligning er relevant fordi tidligere forskning viser at modellprestasjon ofte avhenger av problemstruktur, datagrunnlag og evalueringsoppsett, ikke bare av om modellen er statistisk eller KI-basert (Kolassa, 2022; Schmid, Roidl, Kirchheim, & Pauly, 2025).

For det tredje er den historiske modelltesten tydelig adskilt fra fremtidsprognosene. Dette er viktig fordi det skiller mellom verifiserbar modellprestasjon og ikke-verifiserte fremtidsestimater. I den historiske testen kan modellene vurderes mot faktiske observasjoner ved hjelp av MAE, RMSE, sMAPE og MASE. Fremtidsprognosene kan derimot bare tolkes som modellbaserte estimater, siden faktiske observasjoner ennå ikke foreligger. Dette skillet reduserer risikoen for å tolke fremtidsprognosene som sikre utfall.

Samtidig har studien flere begrensninger. Den viktigste begrensningen er datamaterialets størrelse og struktur. Datasettet dekker en relativt kort tidsperiode, og flere av tidsseriene er preget av mange nullperioder og enkelte ekstreme topper. Dette gjør både modelltrening og evaluering krevende. Korte og nulltunge serier gir begrenset grunnlag for å identifisere stabile trend-, sesong- eller sekvensmønstre, og kan samtidig gjøre modellene følsomme for enkeltobservasjoner. Dette er særlig relevant fordi offhire i materialet ikke opptrer som en jevn utvikling, men som et episodisk fenomen med store variasjoner mellom fartøy. En annen begrensning er at datagrunnlaget består av sekundærdata fra én casebedrift. Selv om dataene er direkte relevante for problemstillingen, kan de ikke verifiseres fullt ut eksternt. Studien er også avgrenset til ett rederi og offshoresegmentet. Funnene kan derfor ikke uten videre generaliseres statistisk til andre rederier, andre fartøytyper eller andre maritime segmenter. Resultatene bør heller forstås som en analytisk og casebasert sammenligning som viser hvordan modellene presterer under de konkrete betingelsene i dette datamaterialet. Det er også en metodisk begrensning at eksterne forklaringsvariabler ikke er inkludert i modellene. Dette er særlig viktig for fremtidsprognosene, der usikkerheten øker jo lengre horisonten blir. Modeller som bare bygger på historisk offhire, har begrenset evne til å fange opp nye markedsendringer eller operasjonelle hendelser som ikke ligner tidligere observasjoner. Videre er det viktig å understreke at studien ikke er en full optimaliseringsstudie av hver enkelt modellfamilie. XGBoost og LSTM er implementert med moderate og relativt faste modelloppsett for å sikre sammenlignbarhet mellom modellene. Dette styrker den kontrollerte sammenligningen, men innebærer samtidig at resultatene ikke nødvendigvis viser den maksimale ytelsen disse modellene kunne oppnådd med mer omfattende hyperparameteroptimalisering, alternative feature-sett eller mer avansert arkitektur. Konklusjonen bør derfor ikke forstås som at KI-baserte modeller generelt er mindre egnet til offhire-prognostisering, men at de i dette konkrete oppsettet og med dette datagrunnlaget ikke ga en tydelig forbedring sammenlignet med SARIMA.

Til slutt er studien prediktiv og ikke kausal. Modellene vurderes etter hvor godt de predikerer månedlig offhire, ikke etter om de forklarer årsakene til offhire. Dette betyr at analysen kan si noe om modellvalg og prediksjonsnøyaktighet, men ikke gi sikre konklusjoner om hvorfor offhire oppstår. Studien er derfor best forstått som en metodisk sammenligning av prognosemodeller brukt som beslutningsstøtte, og ikke som en kausal analyse av tekniske, kommersielle eller operasjonelle årsaker til offhire.

## Anvendelse av prognosene i beslutningsstøtte

For Simon Møkster Shipping AS har funnene først og fremst praktisk verdi som beslutningsstøtte. Studien viser at historiske offhire-data kan brukes til å etablere et mer systematisk grunnlag for å vurdere fremtidig kontraktsutnyttelse, men også at slike prognoser må tolkes med varsomhet. Resultatene peker mot at SARIMA kan fungere som et forsvarlig førstevalg for kortsiktig prognostisering, særlig fordi modellen hadde best historisk testytelse og samtidig er mer transparent enn de KI-baserte modellene. Dette gjør modellen egnet som et mulig basisverktøy for månedlig oppfølging av fartøy med høy historisk offhire.

Den praktiske nytten ligger likevel ikke bare i å velge én modell. Like viktig er det at modellene kan brukes til å synliggjøre usikkerhet. Dersom flere modeller peker mot samme fartøy eller samme periode, kan dette gi et sterkere signal om at videre operasjonell vurdering er nødvendig. Dersom modellene derimot spriker kraftig, slik de gjør på lengre prognosehorisonter, bør dette tolkes som et tegn på høy usikkerhet snarere enn som et grunnlag for bastante beslutninger. På denne måten kan modellene brukes som et supplement til faglig vurderinger, ikke som en erstatning for operasjonell erfaring, kontraktskunnskap eller teknisk vurdering. For bedriften betyr dette at prognosene først og fremst bør anvendes i korte beslutningsvinduer. Prognoser på én til tre måneder kan være relevante for kapasitetsplanlegging, kontraktsoppfølging og prioritering av fartøy som har historikk med høy eller uregelmessig offhire. Prognoser på seks og tolv måneder bør derimot brukes mer forsiktig, særlig fordi modellene ikke inkluderer eksterne forhold som kontraktsendringer, tekniske hendelser eller markedsutvikling. Lange prognosehorisonter kan derfor være nyttige som scenario indikasjoner, men bør ikke behandles som presise forventninger om fremtidig nivå.

En mulig praktisk tilnærming er å bruke SARIMA som basisprognose og samtidig la XGBoost, LSTM og eksponentiell glatting fungere som sammenligningsmodeller. Dersom SARIMA viser økende offhire samtidig som de andre modellene peker i samme retning, kan dette styrke grunnlaget for videre oppfølging. Dersom én modell avviker kraftig fra de andre, bør avviket undersøkes nærmere før det tillegges beslutningsvekt. Slik kan modellene bidra til en mer strukturert vurderingsprosess, der prognoser brukes til å stille bedre spørsmål heller enn å gi endelige svar.

Samlet sett viser studien at modellvalg bør være et eksplisitt beslutningsspørsmål i arbeid med offhire-prognoser. I denne casen fremstår SARIMA som det mest forsvarlige førstevalget basert på historisk testytelse, men resultatene viser også at KI-baserte modeller kan være relevante dersom datagrunnlaget utvides. Den viktigste praktiske implikasjonen er derfor ikke at én modell bør brukes ukritisk, men at prognoser bør inngå som en del av et bredere beslutningsgrunnlag der historiske data, modellresultater og operasjonell kunnskap vurderes samlet.

# Konklusjon

Denne studien viser at valg av prognosemodell påvirker prediksjonsnøyaktigheten for månedlig offhire for fartøy innenfor samme offshoresegment. Samtidig viser resultatene at mer komplekse KI-baserte modeller ikke automatisk gir bedre prognoser enn klassiske tidsseriemodeller. Med utgangspunkt i historiske offhire-data fra Simon Møkster Shipping AS ble SARIMA, eksponentiell glatting, XGBoost og LSTM sammenlignet under samme historiske evalueringsoppsett. Den historiske modelltesten viser at SARIMA ga lavest MAE, RMSE og MASE, og dermed fremstår som det mest treffsikre og forsvarlige førstevalget i denne casen.

Problemstillingen kan derfor besvares med at modellvalg har betydning for prediksjonsnøyaktigheten, men at effekten av modellvalget må forstås i lys av datagrunnlaget. I dette datasettet var det særlig SARIMA, ikke klassiske modeller som gruppe, som ga høyest historisk prediksjonsnøyaktighet. XGBoost og LSTM var konkurransedyktige på absolutte feil, men ga ikke en tydelig forbedring sammenlignet med SARIMA. Eksponentiell glatting fungerte som en nyttig referansemodell, men var samlet svakere på MAE og RMSE. Resultatene tyder dermed på at en relativt transparent fartøyvis tidsseriemodell passet bedre til dette korte, nulltunge og heterogene datamaterialet enn mer fleksible KI-baserte modeller. Sett opp mot det første delspørsmålet viser studien at SARIMA predikerte neste måneds offhire mest presist i den historiske testen. Dette funnet må likevel tolkes som casespesifikt. Det innebærer ikke at SARIMA alltid vil være best for offhire-prognostisering, men at modellen var best egnet blant modellene som ble testet under de konkrete datamessige og metodiske betingelsene i denne oppgaven. Sett opp mot det andre delspørsmålet viser fremtidsprognosene at modellene gir ulike estimater for månedlig offhire på korte og lengre prognosehorisonter. Forskjellene er relativt moderate på kort horisont, men blir tydeligere på 6 og 12 måneders horisont. Dette gjelder særlig XGBoost, som gir betydelig høyere langtidsprognoser enn de øvrige modellene. Slike avvik bør ikke tolkes som sikre fremtidige utfall, men som uttrykk for at modellstruktur og rekursiv prognoselogikk kan få stor betydning når prognosehorisonten forlenges. Fremtidsprognosene bør derfor forstås som beslutningsstøtte og scenario indikasjoner, ikke som presise prediksjoner.

For Simon Møkster Shipping AS ligger den praktiske verdien først og fremst i at historiske offhire-data kan brukes mer systematisk som støtte i kortsiktig planlegging og oppfølging av fartøy. SARIMA kan fungere som en mulig basisprognose, mens avvik mellom modellene kan brukes som signal om økt usikkerhet og behov for nærmere operasjonell vurdering. Prognosene bør likevel ikke brukes som automatiske beslutningsgrunnlag, siden modellene ikke inkluderer eksterne forklaringsvariabler. Oppgavens viktigste bidrag er dermed å vise at modellvalg for offhire-prognoser bør baseres på empirisk testing i den konkrete operasjonelle konteksten, ikke på en generell antakelse om at KI-baserte modeller alltid er bedre. Studien viser også at prognosemodeller kan ha praktisk verdi selv når datagrunnlaget er begrenset, men at resultatene må tolkes med forsiktighet når datasettet er kort, nulltungt og preget av episodiske topper.

Videre forskning bør undersøke om resultatene endrer seg ved bruk av lengre tidsserier, flere fartøy og rikere datagrunnlag. Det vil særlig være relevant å inkludere forklaringsvariabler som kontraktsstatus, tekniske indikatorer, vedlikeholdsdata og markedsforhold. I tillegg bør fremtidsprognosene evalueres når nye faktiske observasjoner foreligger, slik at det kan undersøkes om modellforskjellene består over tid. Fremtidige studier kan også teste mer omfattende hyperparameteroptimalisering og alternative modelloppsett for XGBoost og LSTM, for å vurdere om KI-baserte modeller får større merverdi når datagrunnlaget blir mer informativt.

# Bibliografi

Carbonneau, R., Laframboise, K., & Vahidov, R. (2008). Application of machine learning techniques for supply chain demand forecasting. *European Journal of Operational Research, 184*(3), 1140-1154. https://doi.org/10.1016/j.ejor.2006.12.004

Chen, T., & Guestrin, C. (2016). XGBoost: A scalable tree boosting system. *Proceedings of the 22nd ACM SIGKDD International Conference on Knowledge Discovery and Data Mining* (pp. 785-794). Association for Computing Machinery. https://doi.org/10.1145/2939672.2939785

Chu, Z., Yan, R., & Wang, S. (2024). Vessel turnaround time prediction: A machine learning approach. *Ocean & Coastal Management, 249*, 107021. https://doi.org/10.1016/j.ocecoaman.2024.107021

Douaioui, K., Oucheikh, R., Benmoussa, O., & Mabrouki, C. (2024). Machine learning and deep learning models for demand forecasting in supply chain management: A critical review. *Applied System Innovation, 7*(5), 93. https://doi.org/10.3390/asi7050093

Fildes, R., Kolassa, S., & Ma, S. (2022). Post-script: Retail forecasting: Research and practice. *International Journal of Forecasting, 38*(4), 1319-1324. https://doi.org/10.1016/j.ijforecast.2021.09.012

Gardner, E. S. (1985). Exponential smoothing: The state of the art. *Journal of Forecasting, 4*(1), 1-28. https://doi.org/10.1002/for.3980040103

Hochreiter, S., & Schmidhuber, J. (1997). Long short-term memory. *Neural Computation, 9*(8), 1735-1780. https://doi.org/10.1162/neco.1997.9.8.1735

Hyndman, R. J. (2006). Another look at measures of forecast accuracy. *International Journal of Forecasting*, 22(4), 679-688. https://doi.org/10.1016/j.ijforecast.2006.03.001

Hyndman, R. J., & Athanasopoulos, G. (2021). *Forecasting: principles and practice* (3rd ed.). OTexts. https://otexts.com/fpp3/

Hyndman, R. J., & Khandakar, Y. (2008). Automatic time series forecasting: The forecast package for R. *Journal of Statistical Software, 27*(3), 1-22. https://doi.org/10.18637/jss.v027.i03

Hyndman, R. J., Koehler, A. B., Snyder, R. D., & Grose, S. (2002). A state space framework for automatic forecasting using exponential smoothing methods. *International Journal of Forecasting, 18*(3), 439-454. https://doi.org/10.1016/S0169-2070(01)00110-8

Kalafatelis, A. S., Nomikos, N., Giannopoulos, A., Alexandridis, G., Karditsa, A., & Trakadas, P. (2025). Towards predictive maintenance in the maritime industry: A component-based overview. *Journal of Marine Science and Engineering, 13*(3), 425. https://doi.org/10.3390/jmse13030425

Kjeldsberg, F., & Munim, Z. H. (2024). Automated machine learning driven model for predicting platform supply vessel freight market. *Computers & Industrial Engineering, 191*, 110153. https://doi.org/10.1016/j.cie.2024.110153

Kolassa, S. (2022). Commentary on the M5 forecasting competition. *International Journal of Forecasting, 38*(4), 1562-1568. https://doi.org/10.1016/j.ijforecast.2021.08.006

Ljung, G. M., & Box, G. E. (1978). On a measure of lack of fit in time series models. *Biometrika, 65*(2), 297-303. https://doi.org/10.1093/biomet/65.2.297

Makridakis, S., Spiliotis, E., & Assimakopoulos, V. (2022). M5 accuracy competition: Results, findings, and conclusions. *International Journal of Forecasting, 38*(4), 1346-1364. https://doi.org/10.1016/j.ijforecast.2021.11.013

Menon Economics. (2026). *Maritim verdiskapingsrapport 2026.* Menon Economics. https://menon.no/prosjekter/maritim-verdiskapingsrapport-2026

Schmid, L., Roidl, M., Kirchheim, A., & Pauly, M. (2025). Comparing statistical and machine learning methods for time series forecasting in data-driven logistics: A simulation study. *Entropy, 27*(1), 25. https://doi.org/10.3390/e27010025


# Vedlegg

## Oversikt over figurer

Tabell 14 gir en samlet oversikt over figurene som er brukt i rapporten, hva de viser og hvor de er omtalt.

| Figur | Tittel | Kort beskrivelse | Plassering i rapporten |
|----|----|----|----|
| Figur 1 | Samlet offhire per måned aggregert på tvers av alle fartøy | Viser samlet historisk offhire på tvers av fartøy | 4.0 Casebeskrivelse |
| Figur 2 | Heatmap for offhire per fartøy og måned | Viser variasjon mellom fartøy og over tid | 4.0 Casebeskrivelse |
| Figur 3 | Gjennomsnittlig månedlig offhire per fartøy | Rangerer fartøy etter gjennomsnittlig nivå | 5.2.2 Deskriptiv analyse av datasettet |
| Figur 4 | Boksplott for offhire per fartøy | Viser fordeling, median og ekstreme utslag | 5.2.2 Deskriptiv analyse av datasettet |
| Figur 5 | Tidsserier for fartøy med høyest gjennomsnittlig nedetid | Viser utviklingen for de mest utsatte fartøyene i fem delpaneler | 5.2.2 Deskriptiv analyse av datasettet |
| Figur 6 | ACF for representativ ARIMA/SARIMA-serie | Støtter identifikasjon av representativ SARIMA-modell | 6.1 SARIMA |
| Figur 7 | PACF for representativ ARIMA/SARIMA-serie | Støtter identifikasjon av representativ SARIMA-modell | 6.1 SARIMA |
| Figur 8 | Residualdiagnostikk for representativ ARIMA/SARIMA-modell | Viser residualforløp og residualfordeling | 6.1 SARIMA |
| Figur 9 | Representativ testprognose for ARIMA/SARIMA | Sammenligner testprediksjon med faktisk forløp | 6.1 SARIMA |
| Figur 10 | Representativ testprognose for eksponentiell glatting | Viser modellens testforløp for representativt fartøy | 6.2 Eksponentiell glatting |
| Figur 11 | XGBoost feature importance | Viser hvilke features som betyr mest i modellen | 6.3 XGBoost |
| Figur 12 | Representativ testprognose for XGBoost | Viser modellens testprediksjoner for representativt fartøy | 6.3 XGBoost |
| Figur 13 | Treningshistorikk for LSTM | Viser trenings- og valideringstap over epoker | 6.4 LSTM |
| Figur 14 | Representativ testprognose for LSTM | Viser modellens testprediksjoner for representativt fartøy | 6.4 LSTM |
| Figur 15 | MAE per modell i testperioden | Samlet sammenligning av modellene på MAE | 7.1 Resultater fra historisk modelltesting |
| Figur 16 | MAE per testmåned og modell | Viser hvordan prediksjonsfeilen varierer over tid | 7.1 Resultater fra historisk modelltesting |
| Figur 17 | Heatmap for MAE per fartøy og modell | Viser feilfordeling mellom fartøy og modeller | 7.1 Resultater fra historisk modelltesting |
| Figur 18 | Samlet prognostisert offhire 1 måned fram | Viser én-månedsprognosen på modellnivå | 7.2 Resultater fra fremtidsprognoser |
| Figur 19 | Samlet prognostisert offhire 3 måneder fram | Viser tre-månedersprognosen på modellnivå | 7.2 Resultater fra fremtidsprognoser |
| Figur 20 | Samlet prognostisert offhire 6 måneder fram | Viser seks-månedersprognosen på modellnivå | 7.2 Resultater fra fremtidsprognoser |
| Figur 21 | Samlet prognostisert offhire 12 måneder fram | Viser tolv-månedersprognosen på modellnivå | 7.2 Resultater fra fremtidsprognoser |

## Oversikt over tabeller

Tabell 15 gir en samlet oversikt over tabellene som er brukt i rapporten, hva de viser og hvor de er omtalt.

| Tabell | Tittel | Kort beskrivelse | Plassering i rapporten |
|----|----|----|----|
| Tabell 1 | Datadekning | Oppsummerer tidsperiode, antall observasjoner og datadekning | 5.2.1 Datagrunnlag |
| Tabell 2 | Fartøy med høyest gjennomsnittlig nedetid | Viser topp fem fartøy etter gjennomsnittlig nedetid | 5.2.2 Deskriptiv analyse av datasettet |
| Tabell 3 | Felles evalueringsoppsett | Oppsummerer felles testdesign for modellene | 6.0 Modellering |
| Tabell 4 | Beste kandidatmodeller for representativt fartøy (Fartøy 2) | Viser SARIMA-kandidater rangert etter AIC og BIC | 6.1 SARIMA |
| Tabell 5 | Valgt ETS-spesifikasjon i analysen | Viser fordeling av valgte ETS-varianter | 6.2 Eksponentiell glatting |
| Tabell 6 | XGBoost-featuregrupper | Oppsummerer feature-settet brukt i modellen | 6.3 XGBoost |
| Tabell 7 | XGBoost-hyperparametre | Oppsummerer sentrale hyperparametre | 6.3 XGBoost |
| Tabell 8 | LSTM-oppsett i analysen | Oppsummerer sekvenslengde, inputfeatures og arkitektur | 6.4 LSTM |
| Tabell 9 | Samlet testresultat for modellene | Viser MAE, RMSE, sMAPE og MASE for alle modeller | 7.1 Resultater fra historisk modelltesting |
| Tabell 10 | Samlet prognostisert offhire 1 måned fram | Viser én-månedsprognosen for alle modeller | 7.2 Resultater fra fremtidsprognoser |
| Tabell 11 | Samlet prognostisert offhire 3 måneder fram | Viser tre-månedersprognosen for alle modeller | 7.2 Resultater fra fremtidsprognoser |
| Tabell 12 | Samlet prognostisert offhire 6 måneder fram | Viser seks-månedersprognosen for alle modeller | 7.2 Resultater fra fremtidsprognoser |
| Tabell 13 | Samlet prognostisert offhire 12 måneder fram | Viser tolv-månedersprognosen for alle modeller | 7.2 Resultater fra fremtidsprognoser |

## Datagrunnlag og train/test-splitt

Vedleggene under dokumenterer hvilke data som er brukt, uten å gjengi hele månedsmatrisen i rapporten. Fullstendige numeriske verdier ligger i prosjektfilene som er oppgitt i tabellene. Dette gjør vedlegget kortere og mer lesbart ved utskrift og PDF-eksport.

### Vedleggstabell A1. Oppsummering av datagrunnlag og splitten.

| Element | Verdi |
|----|----|
| Primærkilde | 004 data/raw/Data som skal brukes Anonymisert.csv |
| Treningsfil | 004 data/processed/train.csv |
| Testfil | 004 data/processed/test.csv |
| Observasjonsperiode i rådata | 2021-04 til 2026-03 |
| Treningsperiode for modelltest | 2021-04 til 2024-12 |
| Testperiode for modelltest | 2025-01 til 2026-03 |
| Fartøy i anonymisert rådata | 16 |
| Fartøy i historisk modellsammenligning | 15 |
| Historiske testprediksjoner per modell | 225 |
| Fremtidsprognoser per modell | 180 |

### Vedleggstabell A2. Filregister for datagrunnlag, modellresultater og prognoser.

| Kategori | Fil | GitHub-lenke | Innhold |
|----|----|----|----|
| Datagrunnlag | 004 data/raw/Data som skal brukes Anonymisert.csv | [GitHub](https://github.com/LOG650/G10-julie-individuell/blob/main/004%20data/raw/Data%20som%20skal%20brukes%20Anonymisert.csv) | Full anonymisert månedsmatrise |
| Datagrunnlag | 004 data/processed/train.csv | [GitHub](https://github.com/LOG650/G10-julie-individuell/blob/main/004%20data/processed/train.csv) | Treningsgrunnlaget brukt til modelltilpasning. |
| Datagrunnlag | 004 data/processed/test.csv | [GitHub](https://github.com/LOG650/G10-julie-individuell/blob/main/004%20data/processed/test.csv) | Testgrunnlaget brukt til historisk evaluering. |
| Samlede resultater | 004 data/modeling/outputs/shared/model_comparison_summary.csv | [GitHub](https://github.com/LOG650/G10-julie-individuell/blob/main/004%20data/modeling/outputs/shared/model_comparison_summary.csv) | Samlet modellrangering med MAE, RMSE, sMAPE og MASE. |
| Samlede resultater | 004 data/modeling/outputs/shared/metrics_by_vessel.csv | [GitHub](https://github.com/LOG650/G10-julie-individuell/blob/main/004%20data/modeling/outputs/shared/metrics_by_vessel.csv) | Metrikker per modell og fartøy. |
| Samlede resultater | 004 data/modeling/outputs/shared/predictions.csv | [GitHub](https://github.com/LOG650/G10-julie-individuell/blob/main/004%20data/modeling/outputs/shared/predictions.csv) | Historiske prediksjoner samlet per modell, fartøy og måned. |
| Samlede prognoser | 004 data/modeling/outputs/shared/future_predictions.csv | [GitHub](https://github.com/LOG650/G10-julie-individuell/blob/main/004%20data/modeling/outputs/shared/future_predictions.csv) | Fremtidsprognoser samlet per modell, fartøy, horisont og måned. |
| Samlede prognoser | 004 data/modeling/outputs/shared/future_predictions_12m_pivot.csv | [GitHub](https://github.com/LOG650/G10-julie-individuell/blob/main/004%20data/modeling/outputs/shared/future_predictions_12m_pivot.csv) | Bred 12-måneders prognosetabell per modell og fartøy. |
| SARIMA | 004 data/modeling/outputs/models/SARIMA/predictions.csv | [GitHub](https://github.com/LOG650/G10-julie-individuell/blob/main/004%20data/modeling/outputs/models/SARIMA/predictions.csv) | Historiske SARIMA-prediksjoner per fartøy og måned. |
| SARIMA | 004 data/modeling/outputs/models/SARIMA/future_predictions.csv | [GitHub](https://github.com/LOG650/G10-julie-individuell/blob/main/004%20data/modeling/outputs/models/SARIMA/future_predictions.csv) | SARIMA-fremtidsprognoser per fartøy og måned. |
| SARIMA | 004 data/modeling/outputs/models/SARIMA/modellvalg_per_fartoy.csv | [GitHub](https://github.com/LOG650/G10-julie-individuell/blob/main/004%20data/modeling/outputs/models/SARIMA/modellvalg_per_fartoy.csv) | Valgt SARIMA-spesifikasjon per fartøy. |
| SARIMA | 004 data/modeling/outputs/models/SARIMA/stasjonaritet.csv | [GitHub](https://github.com/LOG650/G10-julie-individuell/blob/main/004%20data/modeling/outputs/models/SARIMA/stasjonaritet.csv) | Stasjonaritetstester brukt i SARIMA-modelleringen. |
| Eksponentiell glatting | 004 data/modeling/outputs/models/Eksponentiell glatting/predictions.csv | [GitHub](https://github.com/LOG650/G10-julie-individuell/blob/main/004%20data/modeling/outputs/models/Eksponentiell%20glatting/predictions.csv) | Historiske ETS-prediksjoner per fartøy og måned. |
| Eksponentiell glatting | 004 data/modeling/outputs/models/Eksponentiell glatting/future_predictions.csv | [GitHub](https://github.com/LOG650/G10-julie-individuell/blob/main/004%20data/modeling/outputs/models/Eksponentiell%20glatting/future_predictions.csv) | ETS-fremtidsprognoser per fartøy og måned. |
| Eksponentiell glatting | 004 data/modeling/outputs/models/Eksponentiell glatting/modellvalg_per_fartoy.csv | [GitHub](https://github.com/LOG650/G10-julie-individuell/blob/main/004%20data/modeling/outputs/models/Eksponentiell%20glatting/modellvalg_per_fartoy.csv) | Valgt ETS-spesifikasjon per fartøy. |
| XGBoost | 004 data/modeling/outputs/models/XGBoost/predictions.csv | [GitHub](https://github.com/LOG650/G10-julie-individuell/blob/main/004%20data/modeling/outputs/models/XGBoost/predictions.csv) | Historiske XGBoost-prediksjoner per fartøy og måned. |
| XGBoost | 004 data/modeling/outputs/models/XGBoost/future_predictions.csv | [GitHub](https://github.com/LOG650/G10-julie-individuell/blob/main/004%20data/modeling/outputs/models/XGBoost/future_predictions.csv) | XGBoost-fremtidsprognoser per fartøy og måned. |
| XGBoost | 004 data/modeling/outputs/models/XGBoost/feature_importance.csv | [GitHub](https://github.com/LOG650/G10-julie-individuell/blob/main/004%20data/modeling/outputs/models/XGBoost/feature_importance.csv) | Feature importance fra XGBoost-modellen. |
| LSTM | 004 data/modeling/outputs/models/LSTM/predictions.csv | [GitHub](https://github.com/LOG650/G10-julie-individuell/blob/main/004%20data/modeling/outputs/models/LSTM/predictions.csv) | Historiske LSTM-prediksjoner per fartøy og måned. |
| LSTM | 004 data/modeling/outputs/models/LSTM/future_predictions.csv | [GitHub](https://github.com/LOG650/G10-julie-individuell/blob/main/004%20data/modeling/outputs/models/LSTM/future_predictions.csv) | LSTM-fremtidsprognoser per fartøy og måned. |
| LSTM | 004 data/modeling/outputs/models/LSTM/training_history.csv | [GitHub](https://github.com/LOG650/G10-julie-individuell/blob/main/004%20data/modeling/outputs/models/LSTM/training_history.csv) | Trenings- og valideringstap per epoch. |

GitHub-lenkene peker til main-branchen i prosjektets repo.

### Vedleggstabell A3. Train/test-splitt per fartøy.

| Fartøy | Trening | Treningsobs. | Test | Testobs. | Modelltest |
|----|----|----|----|----|----|
| Fartøy 1 | 2021-04 til 2024-12 | 45 | 2025-01 til 2026-03 | 15 | Ja |
| Fartøy 2 | 2021-04 til 2024-12 | 45 | 2025-01 til 2026-03 | 15 | Ja |
| Fartøy 3 | 2021-04 til 2024-12 | 45 | 2025-01 til 2026-03 | 15 | Ja |
| Fartøy 4 | 2021-04 til 2024-12 | 45 | 2025-01 til 2026-03 | 15 | Ja |
| Fartøy 5 | 2021-04 til 2024-12 | 45 | 2025-01 til 2026-03 | 15 | Ja |
| Fartøy 6 | 2021-04 til 2024-12 | 45 | 2025-01 til 2026-03 | 15 | Ja |
| Fartøy 7 | 2021-04 til 2024-12 | 45 | 2025-01 til 2026-03 | 15 | Ja |
| Fartøy 8 | 2021-04 til 2024-12 | 45 | 2025-01 til 2026-03 | 15 | Ja |
| Fartøy 9 | 2021-04 til 2024-12 | 45 | 2025-01 til 2026-03 | 15 | Ja |
| Fartøy 10 | 2021-04 til 2024-12 | 45 | 2025-01 til 2026-03 | 15 | Ja |
| Fartøy 11 | 2021-04 til 2024-12 | 45 | 2025-01 til 2026-03 | 15 | Ja |
| Fartøy 12 | 2021-04 til 2024-12 | 45 | 2025-01 til 2026-03 | 15 | Ja |
| Fartøy 13 | 2021-04 til 2024-12 | 45 | 2025-01 til 2026-03 | 15 | Ja |
| Fartøy 14 | 2021-04 til 2024-12 | 45 | 2025-01 til 2026-03 | 15 | Ja |
| Fartøy 15 | 2021-04 til 2024-12 | 45 | 2025-01 til 2026-03 | 15 | Ja |
| Fartøy 16 | Ikke tilgjengelig | 0 | 2026-02 til 2026-03 | 2 | Nei |

Fartøy 16 er med i rådatafilen, men er ikke brukt i den historiske modellsammenligningen fordi fartøyet ikke har tilstrekkelig historikk i treningsperioden.

## Historiske modellresultater per modell

Denne delen beholder modellresultatene som trengs for etterprøvbarhet, men fjerner de fullstendige måned-for-måned-prediksjonstabellene per fartøy. De fullstendige prediksjonene ligger i 004 data/modeling/outputs/shared/predictions.csv og i hver modellmappe under 004 data/modeling/outputs/models/.

*Vedleggstabell B1. Samlet historisk resultat per modell.*

| Modell                 | Testprediksjoner | MAE    | RMSE    | sMAPE    | MASE   |
|------------------------|------------------|--------|---------|----------|--------|
| SARIMA                 | 225              | 6.1521 | 16.7823 | 100.4363 | 0.9098 |
| Eksponentiell glatting | 225              | 8.3676 | 17.9544 | 168.6203 | 1.1775 |
| XGBoost                | 225              | 7.3487 | 17.401  | 182.9827 | 1.2042 |
| LSTM                   | 225              | 7.5722 | 17.4062 | 178.7664 | 1.3198 |

### SARIMA

*Vedleggstabell B2. Historiske testmetrikker per fartøy for SARIMA.*

| Fartøy    | Antall prediksjoner | MAE     | RMSE    | sMAPE    | MASE   |
|-----------|---------------------|---------|---------|----------|--------|
| Fartøy 1  | 15                  | 2.2272  | 3.5343  | 106.6667 | 0.1892 |
| Fartøy 2  | 15                  | 1.2727  | 3.2025  | 93.3333  | 0.0764 |
| Fartøy 3  | 15                  | 2.6236  | 9.2528  | 133.3333 | 1.0809 |
| Fartøy 4  | 15                  | 5.3334  | 20.6559 | 106.6667 | 0.6241 |
| Fartøy 5  | 15                  | 1.2787  | 3.5347  | 53.3333  | 0.0878 |
| Fartøy 6  | 15                  | 0.9297  | 3.5511  | 93.3333  | 0.1515 |
| Fartøy 7  | 15                  | 1.1455  | 1.584   | 152.0805 | 0.15   |
| Fartøy 8  | 15                  | 14.9288 | 24.0346 | 142.836  | 1.046  |
| Fartøy 9  | 15                  | 16.5092 | 32.6134 | 71.9397  | 3.363  |
| Fartøy 10 | 15                  | 27.8226 | 36.3378 | 149.2944 | 1.641  |
| Fartøy 11 | 15                  | 15.5412 | 25.6978 | 111.0294 | 2.4687 |
| Fartøy 12 | 15                  | 0.1154  | 0.4328  | 93.3333  | 0.0465 |
| Fartøy 13 | 15                  | 0       | 0       | 0        |        |
| Fartøy 14 | 15                  | 0       | 0       | 93.3333  | 0      |
| Fartøy 15 | 15                  | 2.5539  | 6.3041  | 106.0307 | 1.8124 |

### Eksponentiell glatting

*Vedleggstabell B3. Historiske testmetrikker per fartøy for Eksponentiell glatting.*

| Fartøy    | Antall prediksjoner | MAE     | RMSE    | sMAPE    | MASE   |
|-----------|---------------------|---------|---------|----------|--------|
| Fartøy 1  | 15                  | 0.3942  | 0.4994  | 200      | 0.0335 |
| Fartøy 2  | 15                  | 10.7607 | 16.5932 | 106.6667 | 0.6459 |
| Fartøy 3  | 15                  | 1.2654  | 1.2895  | 188.3989 | 0.5213 |
| Fartøy 4  | 15                  | 9.507   | 20.1255 | 198.6892 | 1.1125 |
| Fartøy 5  | 15                  | 7.1328  | 7.3454  | 185.0956 | 0.4896 |
| Fartøy 6  | 15                  | 2.9241  | 2.9344  | 200      | 0.4765 |
| Fartøy 7  | 15                  | 4.7339  | 4.8222  | 184.9898 | 0.6199 |
| Fartøy 8  | 15                  | 16.3055 | 24.3851 | 172.9675 | 1.1424 |
| Fartøy 9  | 15                  | 22.6011 | 39.0425 | 175.3113 | 4.6039 |
| Fartøy 10 | 15                  | 31.7437 | 37.2145 | 149.0037 | 1.8723 |
| Fartøy 11 | 15                  | 14.219  | 23.1754 | 169.7775 | 2.2587 |
| Fartøy 12 | 15                  | 1.0583  | 1.062   | 200      | 0.426  |
| Fartøy 13 | 15                  | 0       | 0       | 0        |        |
| Fartøy 14 | 15                  | 0.2517  | 0.2526  | 200      | 0.426  |
| Fartøy 15 | 15                  | 2.6166  | 5.016   | 198.4034 | 1.857  |

### XGBoost

*Vedleggstabell B4. Historiske testmetrikker per fartøy for XGBoost.*

| Fartøy    | Antall prediksjoner | MAE     | RMSE    | sMAPE    | MASE   |
|-----------|---------------------|---------|---------|----------|--------|
| Fartøy 1  | 15                  | 0.9596  | 1.0964  | 200      | 0.0815 |
| Fartøy 2  | 15                  | 1.7336  | 2.6006  | 200      | 0.1041 |
| Fartøy 3  | 15                  | 2.8578  | 8.4898  | 178.9069 | 1.1774 |
| Fartøy 4  | 15                  | 10.2205 | 21.8763 | 199.6813 | 1.196  |
| Fartøy 5  | 15                  | 6.0572  | 7.9921  | 187.3662 | 0.4158 |
| Fartøy 6  | 15                  | 0.9593  | 1.137   | 200      | 0.1563 |
| Fartøy 7  | 15                  | 3.6708  | 9.1184  | 175.483  | 0.4807 |
| Fartøy 8  | 15                  | 13.9513 | 21.9551 | 157.6849 | 0.9775 |
| Fartøy 9  | 15                  | 24.0603 | 39.2209 | 182.8467 | 4.9012 |
| Fartøy 10 | 15                  | 29.1458 | 37.7376 | 130.3543 | 1.7191 |
| Fartøy 11 | 15                  | 11.2642 | 18.7156 | 147.2948 | 1.7893 |
| Fartøy 12 | 15                  | 1.0624  | 1.1872  | 200      | 0.4277 |
| Fartøy 13 | 15                  | 0.5865  | 0.7619  | 200      |        |
| Fartøy 14 | 15                  | 0.8196  | 0.9574  | 200      | 1.387  |
| Fartøy 15 | 15                  | 2.8822  | 6.1128  | 185.1222 | 2.0454 |

### LSTM

*Vedleggstabell B5. Historiske testmetrikker per fartøy for LSTM.*

| Fartøy    | Antall prediksjoner | MAE     | RMSE    | sMAPE    | MASE   |
|-----------|---------------------|---------|---------|----------|--------|
| Fartøy 1  | 15                  | 2.3676  | 3.0475  | 133.3333 | 0.2011 |
| Fartøy 2  | 15                  | 2.0218  | 2.5961  | 186.6667 | 0.1214 |
| Fartøy 3  | 15                  | 2.4692  | 3.1395  | 186.7381 | 1.0173 |
| Fartøy 4  | 15                  | 6.8672  | 20.6693 | 186.5655 | 0.8036 |
| Fartøy 5  | 15                  | 3.3521  | 4.4112  | 183.3148 | 0.2301 |
| Fartøy 6  | 15                  | 1.6981  | 2.0003  | 200      | 0.2767 |
| Fartøy 7  | 15                  | 1.5084  | 1.7677  | 171.4157 | 0.1975 |
| Fartøy 8  | 15                  | 14.8762 | 22.1322 | 181.3943 | 1.0423 |
| Fartøy 9  | 15                  | 25.8024 | 40.8796 | 175.6155 | 5.256  |
| Fartøy 10 | 15                  | 30.1263 | 37.0148 | 153.1417 | 1.7769 |
| Fartøy 11 | 15                  | 15.1522 | 22.3986 | 167.1112 | 2.4069 |
| Fartøy 12 | 15                  | 1.5317  | 1.8253  | 186.6667 | 0.6166 |
| Fartøy 13 | 15                  | 1.4924  | 1.7786  | 186.6667 |        |
| Fartøy 14 | 15                  | 1.4924  | 1.7786  | 186.6667 | 2.5255 |
| Fartøy 15 | 15                  | 2.8255  | 4.7479  | 196.2    | 2.0052 |

### Teknisk modelldokumentasjon

Tabellene under er forkortede støttetabeller for modellvalg og diagnostikk. Fullstendige tekniske resultater ligger i modellmappene under 004 data/modeling/outputs/models/.

*Vedleggstabell B6. Valgt SARIMA-spesifikasjon per fartøy.*

| Fartøy    | Valgt modell               | AIC      | BIC      |
|-----------|----------------------------|----------|----------|
| Fartøy 1  | SARIMA(1,1,2)(0,0,1)\[12\] | 249.1615 | 255.998  |
| Fartøy 2  | SARIMA(2,0,0)(1,0,0)\[12\] | 291.4288 | 297.1648 |
| Fartøy 3  | SARIMA(2,1,0)(1,0,0)\[12\] | 123.86   | 129.4648 |
| Fartøy 4  | SARIMA(0,0,2)(0,0,1)\[12\] | 258.1341 | 263.7388 |
| Fartøy 5  | SARIMA(2,0,1)(0,0,1)\[12\] | 221.95   | 229.12   |
| Fartøy 6  | SARIMA(0,0,2)(1,0,1)\[12\] | 181.0161 | 188.0221 |
| Fartøy 7  | SARIMA(0,1,2)(1,0,1)\[12\] | 225.4706 | 232.3071 |
| Fartøy 8  | SARIMA(2,0,1)(1,0,0)\[12\] | 285.8702 | 293.0402 |
| Fartøy 9  | SARIMA(0,0,2)(0,0,1)\[12\] | 186.5862 | 192.191  |
| Fartøy 10 | SARIMA(2,0,2)(1,0,0)\[12\] | 299.8027 | 308.4066 |
| Fartøy 11 | SARIMA(2,0,0)(1,0,0)\[12\] | 261.0565 | 266.7924 |
| Fartøy 12 | SARIMA(0,0,2)(1,0,1)\[12\] | 196.9529 | 203.9589 |
| Fartøy 13 | Ikke estimert              |          |          |
| Fartøy 14 | SARIMA(0,0,2)(0,0,1)\[12\] | 144.9974 | 150.6021 |
| Fartøy 15 | SARIMA(0,0,2)(0,0,1)\[12\] | 198.2601 | 203.8649 |

*Vedleggstabell B7. Oppsummert SARIMA-stasjonaritetstest.*

| Testvariant               | Stasjonære | Ikke stasjonære | Antall tester |
|---------------------------|------------|-----------------|---------------|
| Ingen differensiering     | 11         | 3               | 14            |
| Første differense         | 14         | 0               | 14            |
| Sesongdifferense (12)     | 11         | 3               | 14            |
| Første + sesongdifferense | 12         | 2               | 14            |

*Vedleggstabell B8. Oppsummert SARIMA-residualdiagnostikk.*

| Diagnostikk | Antall fartøy | Tolkning |
|----|----|----|
| Ljung-Box p-verdi \>= 0,05 | 14 | Ingen tydelig restautokorrelasjon etter valgt grense. |
| Ljung-Box p-verdi \< 0,05 | 0 | Indikerer mulig restautokorrelasjon. |
| Ikke estimert | 1 | Manglende modellgrunnlag for fartøyet. |

*Vedleggstabell B9. Valgt eksponentiell glatting-spesifikasjon per fartøy.*

| Fartøy    | Valgt modell | AIC      | BIC      |
|-----------|--------------|----------|----------|
| Fartøy 1  | ANN          | 250.6617 | 254.275  |
| Fartøy 2  | AAA          | 299.2254 | 328.132  |
| Fartøy 3  | ANN          | 163.315  | 166.9284 |
| Fartøy 4  | ANN          | 245.305  | 248.9183 |
| Fartøy 5  | ANN          | 281.109  | 284.7223 |
| Fartøy 6  | ANN          | 245.2361 | 248.8495 |
| Fartøy 7  | ANN          | 269.8291 | 273.4424 |
| Fartøy 8  | ANN          | 284.787  | 288.4003 |
| Fartøy 9  | ANN          | 232.9739 | 236.5872 |
| Fartøy 10 | ANN          | 294.2332 | 297.8465 |
| Fartøy 11 | ANN          | 230.2082 | 233.8215 |
| Fartøy 12 | ANN          | 141.6678 | 145.2811 |
| Fartøy 13 | CONST        |          |          |
| Fartøy 14 | ANN          | 62.5344  | 66.1477  |
| Fartøy 15 | ANN          | 163.6983 | 167.3116 |

*Vedleggstabell B10. Topp 10 XGBoost-feature importance.*

| Feature                | Importance |
|------------------------|------------|
| num\_\_lag_1           | 0.1065     |
| cat\_\_vessel_Fartøy 2 | 0.0753     |
| num\_\_rolling_mean_12 | 0.0639     |
| num\_\_rolling_mean_6  | 0.0583     |
| num\_\_rolling_mean_3  | 0.0581     |
| num\_\_time_idx        | 0.0496     |
| num\_\_lag_6           | 0.045      |
| num\_\_rolling_std_6   | 0.0436     |
| cat\_\_vessel_Fartøy 5 | 0.0401     |
| num\_\_rolling_std_12  | 0.0395     |

*Vedleggstabell B11. Kompakt LSTM-treningshistorikk.*

| Punkt                | Epoch | Loss   | Val_loss |
|----------------------|-------|--------|----------|
| Første epoch         | 1     | 0.9635 | 1.1229   |
| Beste valideringstap | 4     | 0.8789 | 1.0326   |
| Siste epoch          | 14    | 0.785  | 1.2213   |


## Fremtidsprognoser

Fremtidsprognosene i rapporten er punktprognoser for 2026-04 til 2027-03. Vedlegget viser en kompakt total per måned og modell. Fullstendige prognoser per fartøy beholdes som CSV-filer i modellutdataene, ikke som lange Word-tabeller.

*Vedleggstabell C1. Samlet 12-måneders fremtidsprognose per måned og modell.*

| Dato    | SARIMA | Eksponentiell glatting | XGBoost | LSTM  |
|---------|--------|------------------------|---------|-------|
| 2026-04 | 76.35  | 53.37                  | 102.19  | 91.6  |
| 2026-05 | 138.96 | 53.69                  | 181.85  | 77.58 |
| 2026-06 | 154.27 | 54                     | 238.57  | 66.39 |
| 2026-07 | 113.85 | 54.32                  | 191.26  | 50.8  |
| 2026-08 | 125.77 | 54.63                  | 155.66  | 41.29 |
| 2026-09 | 58.83  | 54.95                  | 206.98  | 25.67 |
| 2026-10 | 16.84  | 55.26                  | 310.44  | 24.81 |
| 2026-11 | 27.08  | 55.58                  | 363.72  | 39.8  |
| 2026-12 | 14.65  | 55.89                  | 416.52  | 54.01 |
| 2027-01 | 121.88 | 56.21                  | 522.01  | 66.39 |
| 2027-02 | 149.42 | 56.52                  | 547.73  | 67.61 |
| 2027-03 | 127.6  | 56.84                  | 353.12  | 64.86 |

Fullstendige prediksjons- og prognosefiler er samlet i filregisteret i vedleggstabell A2.

## Kodevedlegg

Kodevedleggene nedenfor viser modellspesifikke funksjonsuttrekk fra implementasjonen som er brukt i studien. Hensikten er å dokumentere hvordan hver modell er implementert, uten å gjengi hele kodebasen i vedlegget.

### SARIMA-kode

Vedlegg 11.6.1 viser funksjonen run_sarima, som står for fartøyvis modellvalg, residualdiagnostikk og ekspanderende 1-stegs prediksjon for ARIMA/SARIMA.

    def run_sarima(
        train_panel: pd.DataFrame,
        test_panel: pd.DataFrame,
        split_metadata: dict[str, Any],
    ) -> tuple[ModelResult, pd.DataFrame]:
        prediction_rows: list[dict[str, Any]] = []
        stationarity_rows: list[dict[str, Any]] = []
        model_rows: list[dict[str, Any]] = []
        residual_rows: list[dict[str, Any]] = []
        representative_vessel = select_representative_vessel(train_panel)
        representative_candidates = pd.DataFrame()
        representative_transformed: pd.Series | None = None
        representative_residuals: pd.Series | None = None
        fallback_representative: str | None = None

        train_vessels = sorted(set(train_panel["vessel"]).intersection(test_panel["vessel"]))
        for vessel in train_vessels:
            train_series = build_vessel_series(train_panel, vessel).dropna()
            test_series = build_vessel_series(test_panel, vessel).dropna()
            if len(train_series) < 24 or test_series.empty:
                continue

            if train_series.nunique() <= 1:
                constant_value = float(train_series.iloc[-1])
                model_rows.append(
                    {
                        "vessel": vessel,
                        "p": np.nan,
                        "d": np.nan,
                        "q": np.nan,
                        "P": np.nan,
                        "D": np.nan,
                        "Q": np.nan,
                        "s": 0,
                        "aic": np.nan,
                        "bic": np.nan,
                        "train_observations": int(len(train_series)),
                    }
                )
                residual_rows.append(
                    {
                        "vessel": vessel,
                        "ljung_box_lag": np.nan,
                        "ljung_box_pvalue": np.nan,
                    }
                )
                for date_value, actual in test_series.items():
                    prediction_rows.append(
                        {
                            "model": "sarima",
                            "vessel": vessel,
                            "date": pd.Timestamp(date_value).strftime("%Y-%m-%d"),
                            "actual": float(actual),
                            "prediction": constant_value,
                        }
                    )
                continue

            selected_d, selected_D, transformed_series, stationarity_results = select_sarima_differencing(
                train_series
            )
            stationarity_rows.extend(
                {"vessel": vessel, **row} for row in stationarity_results
            )
            candidate_df = fit_sarima_candidates(train_series, d=selected_d, D=selected_D)
            best_candidate = candidate_df.iloc[0]
            selected_order = (
                int(best_candidate["p"]),
                int(best_candidate["d"]),
                int(best_candidate["q"]),
            )
            selected_seasonal_order = (
                int(best_candidate["P"]),
                int(best_candidate["D"]),
                int(best_candidate["Q"]),
                int(best_candidate["s"]),
            )

            final_model = SARIMAX(
                train_series,
                order=selected_order,
                seasonal_order=selected_seasonal_order,
                enforce_stationarity=False,
                enforce_invertibility=False,
            )
            final_fit = final_model.fit(disp=False)
            residuals = pd.Series(final_fit.resid, index=train_series.index).dropna()
            ljung_lag = min(12, max(len(residuals) // 2, 1))
            ljung_box = acorr_ljungbox(residuals, lags=[ljung_lag], return_df=True)

            model_rows.append(
                {
                    "vessel": vessel,
                    "p": selected_order[0],
                    "d": selected_order[1],
                    "q": selected_order[2],
                    "P": selected_seasonal_order[0],
                    "D": selected_seasonal_order[1],
                    "Q": selected_seasonal_order[2],
                    "s": selected_seasonal_order[3],
                    "aic": float(best_candidate["aic"]),
                    "bic": float(best_candidate["bic"]),
                    "train_observations": int(len(train_series)),
                }
            )
            residual_rows.append(
                {
                    "vessel": vessel,
                    "ljung_box_lag": int(ljung_lag),
                    "ljung_box_pvalue": float(ljung_box["lb_pvalue"].iloc[0]),
                }
            )

            history = train_series.copy()
            for date_value, actual in test_series.items():
                prediction = float(
                    fit_or_fallback_sarima_forecast(
                        history,
                        1,
                        order=selected_order,
                        seasonal_order=selected_seasonal_order,
                    )[0]
                )
                prediction_rows.append(
                    {
                        "model": "sarima",
                        "vessel": vessel,
                        "date": pd.Timestamp(date_value).strftime("%Y-%m-%d"),
                        "actual": float(actual),
                        "prediction": prediction,
                    }
                )
                history = pd.concat(
                    [history, pd.Series([float(actual)], index=pd.DatetimeIndex([date_value]))]
                )

            if fallback_representative is None:
                fallback_representative = vessel
            if representative_vessel == vessel or (
                representative_vessel not in train_vessels and fallback_representative == vessel
            ):
                representative_candidates = candidate_df.head(10).copy()
                representative_transformed = transformed_series
                representative_residuals = residuals
                representative_vessel = vessel

        if representative_candidates.empty and fallback_representative is not None:
            representative_vessel = fallback_representative

        if not prediction_rows:
            return (
                ModelResult(
                    model="sarima",
                    status="skipped",
                    details={
                        **split_metadata,
                        "reason": "Ingen fartøy hadde tilstrekkelig historikk og variasjon til ARIMA/SARIMA.",
                    },
                ),
                pd.DataFrame(),
            )

        pred_df = pd.DataFrame(prediction_rows)
        write_dataframe_artifacts(
            pd.DataFrame(stationarity_rows),
            model_artifact_path("sarima", "stasjonaritet.csv"),
            "ARIMA/SARIMA stasjonaritet per fartøy",
        )
        write_dataframe_artifacts(
            representative_candidates,
            model_artifact_path("sarima", "kandidatmodeller.csv"),
            "ARIMA/SARIMA kandidatmodeller for representativt fartøy",
        )
        write_dataframe_artifacts(
            pd.DataFrame(model_rows),
            model_artifact_path("sarima", "modellvalg_per_fartoy.csv"),
            "ARIMA/SARIMA modellvalg per fartøy",
        )
        write_dataframe_artifacts(
            pd.DataFrame(residual_rows),
            model_artifact_path("sarima", "residualdiagnostikk.csv"),
            "ARIMA/SARIMA residualdiagnostikk per fartøy",
        )
        if (
            representative_vessel is not None
            and representative_transformed is not None
            and representative_residuals is not None
        ):
            save_sarima_diagnostic_plots(
                representative_vessel,
                representative_transformed,
                representative_residuals,
            )
        save_representative_prediction_plot(
            "sarima",
            representative_vessel,
            train_panel,
            test_panel,
            pred_df,
            "ARIMA/SARIMA: representativt testforløp",
        )

        mae_value, rmse_value, smape_value = summarize_prediction_frame(pred_df)
        metrics = ModelResult(
            model="sarima",
            status="ok",
            mae=mae_value,
            rmse=rmse_value,
            smape=smape_value,
            details={
                **split_metadata,
                "series_type": "per_vessel",
                "test_rows": int(len(pred_df)),
                "evaluation_method": "ekspanderende 1-stegs prognose per fartøy",
                "evaluation_level": "fartøynivå",
                "modeled_vessels": int(pred_df["vessel"].nunique()),
                "representative_vessel": representative_vessel,
                "artifact_files": {
                    "stasjonaritet": "stasjonaritet.md",
                    "kandidatmodeller": "kandidatmodeller.md",
                    "modellvalg_per_fartoy": "modellvalg_per_fartoy.md",
                    "residualdiagnostikk_tabell": "residualdiagnostikk.md",
                    "acf": "acf.png",
                    "pacf": "pacf.png",
                    "residualdiagnostikk_figur": "residualdiagnostikk.png",
                    "representativ_testplot": "representativ_testplot.png",
                },
            },
        )
        return metrics, pred_df

### Eksponentiell glatting-kode

Vedlegg 11.6.2 viser funksjonen run_exponential_smoothing, som står for fartøyvis modellvalg mellom ETS-varianter og ekspanderende 1-stegs prediksjon gjennom testperioden.

    def run_exponential_smoothing(
        train_panel: pd.DataFrame,
        test_panel: pd.DataFrame,
        split_metadata: dict[str, Any],
    ) -> tuple[ModelResult, pd.DataFrame]:
        predictions: list[dict[str, Any]] = []
        model_selection_rows: list[dict[str, Any]] = []
        residual_rows: list[dict[str, Any]] = []
        representative_vessel = select_representative_vessel(train_panel)

        for vessel, test_vessel_df in test_panel.groupby("vessel"):
            train_series = build_vessel_series(train_panel, vessel).dropna()
            test_series = build_vessel_series(test_panel, vessel).dropna()
            if len(train_series) < 12 or test_series.empty:
                continue

            if train_series.nunique() <= 1:
                constant_value = float(train_series.iloc[-1])
                model_selection_rows.append(
                    {
                        "vessel": vessel,
                        "spec": "CONST",
                        "aic": np.nan,
                        "bic": np.nan,
                        "train_observations": int(len(train_series)),
                    }
                )
                residual_rows.append(
                    {
                        "vessel": vessel,
                        "ljung_box_lag": np.nan,
                        "ljung_box_pvalue": np.nan,
                    }
                )
                for date_value, actual in test_series.items():
                    predictions.append(
                        {
                            "model": "exponential_smoothing",
                            "vessel": vessel,
                            "date": pd.Timestamp(date_value).strftime("%Y-%m-%d"),
                            "actual": float(actual),
                            "prediction": constant_value,
                        }
                    )
                continue

            candidate_df = fit_ets_candidates(train_series)
            best_candidate = candidate_df.iloc[0]
            fit = fit_ets_model(
                train_series,
                trend=normalize_optional_string(best_candidate["trend"]),
                seasonal=normalize_optional_string(best_candidate["seasonal"]),
                seasonal_periods=(
                    int(best_candidate["seasonal_periods"])
                    if pd.notna(best_candidate["seasonal_periods"])
                    else None
                ),
            )
            residuals = pd.Series(fit.resid, index=train_series.index).dropna()
            ljung_lag = min(12, max(len(residuals) // 2, 1))
            ljung_box = acorr_ljungbox(residuals, lags=[ljung_lag], return_df=True)

            model_selection_rows.append(
                {
                    "vessel": vessel,
                    "spec": best_candidate["spec"],
                    "aic": float(best_candidate["aic"]),
                    "bic": float(best_candidate["bic"]),
                    "train_observations": int(len(train_series)),
                }
            )
            residual_rows.append(
                {
                    "vessel": vessel,
                    "ljung_box_lag": int(ljung_lag),
                    "ljung_box_pvalue": float(ljung_box["lb_pvalue"].iloc[0]),
                }
            )

            history = train_series.copy()
            for date_value, actual in test_series.items():
                pred = float(
                    fit_or_fallback_exponential_forecast(
                        history,
                        1,
                        trend=normalize_optional_string(best_candidate["trend"]),
                        seasonal=normalize_optional_string(best_candidate["seasonal"]),
                        seasonal_periods=(
                            int(best_candidate["seasonal_periods"])
                            if pd.notna(best_candidate["seasonal_periods"])
                            else None
                        ),
                    )[0]
                )
                predictions.append(
                    {
                        "model": "exponential_smoothing",
                        "vessel": vessel,
                        "date": pd.Timestamp(date_value).strftime("%Y-%m-%d"),
                        "actual": float(actual),
                        "prediction": pred,
                    }
                )
                history = pd.concat(
                    [history, pd.Series([float(actual)], index=pd.DatetimeIndex([date_value]))]
                )

        if not predictions:
            return (
                ModelResult(
                    model="exponential_smoothing",
                    status="skipped",
                    details={
                        **split_metadata,
                        "reason": "For få observasjoner per fartøy til å kjøre eksponentiell glatting.",
                    },
                ),
                pd.DataFrame(),
            )

        pred_df = pd.DataFrame(predictions)
        write_dataframe_artifacts(
            pd.DataFrame(model_selection_rows),
            model_artifact_path("exponential_smoothing", "modellvalg_per_fartoy.csv"),
            "Eksponentiell glatting modellvalg per fartøy",
        )
        write_dataframe_artifacts(
            pd.DataFrame(residual_rows),
            model_artifact_path("exponential_smoothing", "residualdiagnostikk.csv"),
            "Eksponentiell glatting residualdiagnostikk",
        )
        save_representative_prediction_plot(
            "exponential_smoothing",
            representative_vessel,
            train_panel,
            test_panel,
            pred_df,
            "Eksponentiell glatting: representativt testforløp",
        )
        mae_value, rmse_value, smape_value = summarize_prediction_frame(pred_df)
        metrics = ModelResult(
            model="exponential_smoothing",
            status="ok",
            mae=mae_value,
            rmse=rmse_value,
            smape=smape_value,
            details={
                **split_metadata,
                "vessels_used": int(pred_df["vessel"].nunique()),
                "test_rows": int(len(pred_df)),
                "evaluation_method": "ekspanderende 1-stegs prognose gjennom testperioden",
                "evaluation_level": "fartøynivå",
                "representative_vessel": representative_vessel,
                "selection_table": "modellvalg_per_fartoy.md",
                "residual_table": "residualdiagnostikk.md",
            },
        )
        return metrics, pred_df

### XGBoost-kode

Vedlegg 11.6.3 viser funksjonen run_xgboost, som står for feature-basert panelmodellering og ekspanderende 1-stegs prediksjon med månedlig re-trening.

    def run_xgboost(
        panel_df: pd.DataFrame,
        split_metadata: dict[str, Any],
    ) -> tuple[ModelResult, pd.DataFrame]:
        history_panel = panel_df[panel_df["date"] <= TRAIN_END_DATE].copy()
        test_panel = panel_df[panel_df["date"] >= TEST_START_DATE].copy()
        reference_train_df = build_panel_features(history_panel)
        if reference_train_df.empty:
            raise DataTooShortError("For få historiske observasjoner til å bygge XGBoost-features.")

        reference_pipeline, feature_columns = build_xgboost_pipeline()
        reference_pipeline.fit(reference_train_df[feature_columns], reference_train_df["offhire_days"])
        save_feature_importance_artifacts(reference_pipeline)

        representative_vessel = select_representative_vessel(history_panel)
        predictions: list[dict[str, Any]] = []
        test_dates = sorted(test_panel["date"].unique())

        for date_value in test_dates:
            train_df = build_panel_features(history_panel)
            if train_df.empty:
                continue
            target_month_df = test_panel[test_panel["date"] == date_value].copy()
            feature_rows = build_xgboost_feature_rows(history_panel, target_month_df)
            if feature_rows.empty:
                history_panel = pd.concat([history_panel, target_month_df], ignore_index=True)
                history_panel = history_panel.sort_values(["vessel", "date"]).reset_index(drop=True)
                continue

            pipeline, feature_columns = build_xgboost_pipeline()
            pipeline.fit(train_df[feature_columns], train_df["offhire_days"])
            step_predictions = np.clip(
                pipeline.predict(feature_rows[feature_columns]),
                0.0,
                MAX_OFFHIRE_VALUE,
            )
            for _, row, prediction in zip(feature_rows.index, feature_rows.itertuples(index=False), step_predictions):
                predictions.append(
                    {
                        "model": "xgboost",
                        "vessel": row.vessel,
                        "date": pd.Timestamp(row.date).strftime("%Y-%m-%d"),
                        "actual": float(row.actual),
                        "prediction": float(prediction),
                    }
                )

            history_panel = pd.concat([history_panel, target_month_df], ignore_index=True)
            history_panel = history_panel.sort_values(["vessel", "date"]).reset_index(drop=True)

        pred_df = pd.DataFrame(predictions)
        save_representative_prediction_plot(
            "xgboost",
            representative_vessel,
            panel_df[panel_df["date"] <= TRAIN_END_DATE],
            test_panel,
            pred_df,
            "XGBoost: representativt testforløp",
        )
        mae_value, rmse_value, smape_value = summarize_prediction_frame(pred_df)
        metrics = ModelResult(
            model="xgboost",
            status="ok",
            mae=mae_value,
            rmse=rmse_value,
            smape=smape_value,
            details={
                **split_metadata,
                "train_rows": int(len(reference_train_df)),
                "test_rows": int(len(pred_df)),
                "walk_forward_steps": int(len(test_dates)),
                "evaluation_method": "ekspanderende 1-stegs prognose med månedlig re-trening",
                "evaluation_level": "fartøynivå",
                "feature_columns": feature_columns,
                "representative_vessel": representative_vessel,
                "model_hyperparameters": {
                    "n_estimators": 200,
                    "max_depth": 4,
                    "learning_rate": 0.05,
                    "subsample": 0.9,
                    "colsample_bytree": 0.9,
                },
            },
        )
        return metrics, pred_df

### LSTM-kode

Vedlegg 11.6.4 viser funksjonen run_lstm, som står for sekvensbygging, månedlig re-trening og ekspanderende 1-stegs prediksjon for den globale LSTM-modellen.

    def run_lstm(
        panel_df: pd.DataFrame,
        split_metadata: dict[str, Any],
    ) -> tuple[ModelResult, pd.DataFrame]:
        history_panel = panel_df[panel_df["date"] <= TRAIN_END_DATE].copy()
        test_panel = panel_df[panel_df["date"] >= TEST_START_DATE].copy()
        representative_vessel = select_representative_vessel(history_panel)
        window_size = 12

        X_reference, y_reference, _ = build_lstm_sequences(history_panel, window_size=window_size)
        _, _, _, history_df = train_lstm_regressor(X_reference, y_reference)
        save_lstm_training_history(history_df)

        pred_records: list[dict[str, Any]] = []
        test_dates = sorted(test_panel["date"].unique())

        for date_value in test_dates:
            X_train, y_train, _ = build_lstm_sequences(history_panel, window_size=window_size)
            if len(X_train) == 0:
                continue

            model, x_scaler, y_scaler, _ = train_lstm_regressor(X_train, y_train)
            month_df = test_panel[test_panel["date"] == date_value].copy()

            for _, row in month_df.iterrows():
                vessel_history = (
                    history_panel[history_panel["vessel"] == row["vessel"]]
                    .sort_values("date")
                    .reset_index(drop=True)
                )
                if len(vessel_history) < window_size:
                    continue

                history_values = vessel_history["offhire_days"].to_numpy(dtype=np.float32)[-window_size:]
                history_months = vessel_history["month_num"].to_numpy(dtype=np.float32)[-window_size:]
                history_special = (
                    vessel_history["Spesielle behov/krav"]
                    .fillna("")
                    .str.strip()
                    .ne("")
                    .astype(np.float32)
                    .to_numpy()[-window_size:]
                )
                sequence = np.stack(
                    [
                        history_values,
                        np.sin(2 * np.pi * history_months / 12.0),
                        np.cos(2 * np.pi * history_months / 12.0),
                        history_special,
                    ],
                    axis=1,
                )
                sequence_scaled = x_scaler.transform(sequence).reshape(1, sequence.shape[0], sequence.shape[1])
                prediction_scaled = model.predict(sequence_scaled, verbose=0).reshape(-1, 1)
                prediction = float(
                    np.clip(
                        y_scaler.inverse_transform(prediction_scaled).reshape(-1)[0],
                        0.0,
                        MAX_OFFHIRE_VALUE,
                    )
                )
                pred_records.append(
                    {
                        "model": "lstm",
                        "vessel": row["vessel"],
                        "date": pd.Timestamp(row["date"]).strftime("%Y-%m-%d"),
                        "actual": float(row["offhire_days"]),
                        "prediction": prediction,
                    }
                )

            history_panel = pd.concat([history_panel, month_df], ignore_index=True)
            history_panel = history_panel.sort_values(["vessel", "date"]).reset_index(drop=True)

        pred_df = pd.DataFrame(pred_records)
        save_representative_prediction_plot(
            "lstm",
            representative_vessel,
            panel_df[panel_df["date"] <= TRAIN_END_DATE],
            test_panel,
            pred_df,
            "LSTM: representativt testforløp",
        )
        mae_value, rmse_value, smape_value = summarize_prediction_frame(pred_df)
        metrics = ModelResult(
            model="lstm",
            status="ok",
            mae=mae_value,
            rmse=rmse_value,
            smape=smape_value,
            details={
                **split_metadata,
                "train_sequences": int(len(X_reference)),
                "test_sequences": int(len(pred_df)),
                "evaluation_method": "ekspanderende 1-stegs prognose med månedlig re-trening",
                "evaluation_level": "fartøynivå",
                "walk_forward_steps": int(len(test_dates)),
                "sequence_length": window_size,
                "input_features": ["offhire_days", "month_sin", "month_cos", "special_flag"],
                "representative_vessel": representative_vessel,
                "architecture": {
                    "lstm_units": 32,
                    "dense_units": 16,
                    "batch_size": 8,
                    "max_epochs": 100,
                },
            },
        )
        return metrics, pred_df
