## Projekt
Dieses Projekt enthält das Python-Programm zur Auswertung der in dem Fachmodul MCI durchgeführten Studie und die Ergebnisse.

## Getting Started

### Prerequisites

python == ^3.10.2
pip == ^22.3.1

### Installation

Die Ausführung des Programmes besteht aus folgenden Schritten

1. Git bash oder ein gleichwertiges Terminal öffnen
2. Installation einer virtuellen Umgebung mit pip
   ```
   pip install virtualenv
   ```
3. Erstellend er virtuellen Umgebung
   ```
   virtualenv venv
   ```
4. Aktivieren der virtuellen Umgebung
   ```
   source venv/Scripts/activate
   ```
5. Installation aller benötigten Pakete zur Ausführung des Programmes in die virtuelle Umgebung
   ```
   pip install -r requirements.txt
   ```

5. Ausführen des Programmes entwerder über die IDE oder mittels
   ```
   python main.py
   ```
### Ordnerstruktur:

Die main.py ist der Ausgangspunkt zur Ausführung des Programmes.
In der analysis.py sind alle Funktionen zur statistischen Auswertung ausprogrammiert:
- Daten einlesen
- Generierung der DataFrames zur späteren Auswertung
- Unabhängige T-Tests
- Einfaktorielle ANOVA
- Korrelationsanalyse

Der Ordner data enthält die Rohdaten in Form von CSV Dateien.

Der Ordner statistsics beinhalten:
- time_precision_boxplot.png, den Boxplot zu den physikalischen Metriken Zeit und Genauigkeit
- questionnaire_barplots, die Balkendiagramme zur Darstellung der Mittelwerte der Komponenten aus dem Fragebogen
- questionnaire_boxplots, den Boxplot zur Darstellung der Komponenten aus dem Fragebogen
- correlation_plot.png, eine Darstellung der Korrelationen der physikalischen Messwerte mit den Komponenten des Fragebogens als Heatmap
- statistics.txt, die strukturierte Ausgabe des Programmes mit allen relevanten Messwerten und Statistiken