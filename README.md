### Dataset referentie

Dataset #2: Stock dataset Polygon.io voor AAPL of de beurs van Apple
[link: polygon.io](https://polygon.io/)

## MLOps Workflow voor AAPL Stock Prediction met Polygon.io API

### Dataset

#### Gebruikte data

Voor dit project maken we gebruik van historische en realtime gegevens over de aandelen van AAPL (Apple Inc.), verkregen via de Polygon.io API. De dataset omvat onder andere historische prijzen, handelsvolume, markttrends en andere relevante gegevens die nodig zijn voor het voorspellen van de aandelenkoersen van AAPL.

#### Data splitsing

We zullen de beschikbare dataset opsplitsen in training, validatie en testsets. De trainingset zal worden gebruikt om het voorspellingsmodel te kunnen trainen, terwijl de validatieset zal worden gebruikt om de prestaties van het model te valideren. De testset zal worden gebruikt om de prestaties van het model te evalueren.

#### Toegang tot nieuwe data

Om nieuwe gegevens te verkrijgen en onze service te blijven voeden met data, zullen we regelmatig verzoek sturen naar de Polygon.io API om de meest recente gegevens over AAPL-aandelen op te halen. Dit proces zal worden geautomatiseerd als onderdeel van onze mlops-workflow.

### Projectuitleg

#### Wat doet de service?

De service voert voorspellingen uit over de aandelenkoersen van AAPL op basis van historische gegevens en markt-data. Het doel van dit project is om een nauwkeurig voorspellingsmodel te ontwikkelen dat investeerders kan helpen bij het nemen van beslissingen op de aandelenmarkt bijvoorbeeld.

#### Toepassing

Dit project resulteert in een applicatie die voorziet van voorspellingen over de toekomstige prijzen van AAPL-aandelen. Deze applicatie kan worden gebruikt door individuele beleggers, om hun investeringsbeslissingen te informeren en of te ondersteunen.

### Stroom & Actie

#### Welke stromen & acties zijn vereist?

1. **Dataverkrijging**: Verzoek versturen naar de Polygon.io API om historische en de laatste gegevens over AAPL-aandelen op te halen.

2. **Dataverwerking**: Verwerken van de verkregen gegevens en deze verdelen in training, validatie en testsets.

3. **Modeltraining**: Trainen van een voorspellingsmodel met behulp van machine learning technieken zoals bijvoorbeeld LSTM (Long Short-Term Memory)/RNN (Recurrent neural network) en ARIMA (AutoRegressive Integrated Moving Average).

4. **Modelvalidatie**: Valideren van het getrainde model met behulp van de validatieset en finetunen waar nodig.

5. **Modelevaluatie**: Evalueren van het model met behulp van de testset om de nauwkeurigheid/prestaties te beoordelen.

6. **Voorspellingen genereren**: Genereren van voorspellingen over de toekomstige prijzen van AAPL-aandelen op basis van het getrainde model en nieuwe gegevens van de Polygon.io API.

7. **Service implementatie**: Implementatie van de voorspellingsdienst als een service met behulp van tools zoals MLflow en Prefect waarmee continu nieuwe gegevens kunnen worden verwerkt en investeerders van actuele voorspellingen kunnen worden voorzien.

Bers Goudantov
2ITAI
