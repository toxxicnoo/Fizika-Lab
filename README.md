# Fizika Laboratorijska Aplikacija

Ova aplikacija je namenjena za analizu laboratorijskih podataka u fizici. Omogućava učitavanje Excel fajlova sa podacima merenja, izračunavanje regresije, prikaz grafikona i generisanje PDF izveštaja.

## Instalacija

1. Preuzmite `fizikaapp.exe` sa sajta (https://github.com/toxxicnoo/Fizika-Lab).
2. Pokrenite `fizikaapp.exe` (nema virusa, aplikacija je pravljena od strane ucenika).

Alternativno, ako želite da pokrenete iz izvornog koda:

1. Instalirajte Python 3.8+.
2. Klonirajte repo: `git clone https://github.com/toxxicnoo/Aplikacija-FIZIKA.git`
3. Instalirajte zavisnosti: `pip install -r requirements.txt`
4. Pokrenite: `python main.py`

## Korišćenje Aplikacije

### 1. Pokretanje Aplikacije

- Dvokliknite na `fizikaapp.exe` ili pokrenite `python main.py`.
- Otvoriće se glavni prozor aplikacije.

### 2. Učitavanje Podataka

- Kliknite na "Učitaj Excel" dugme.
- Izaberite Excel fajl (.xlsx) sa podacima merenja.
- Podaci će se prikazati u tabeli.

### 3. Priprema Podataka za Regresiju

- U desnom panelu, označite kolone koje želite da uključite u PDF izveštaj.
- Za regresiju, izaberite x i y kolone iz padajućih menija.
- Kliknite "Izračunaj Regresiju" da vidite jednačinu i R² vrednost.

### 4. Prikaz Grafa

- Grafikon se automatski ažurira nakon učitavanja podataka i regresije.
- Možete zumirati, pomerati grafikon itd.

### 5. Generisanje PDF Izveštaja

- Kliknite "Generiši PDF" dugme.
- Izaberite lokaciju za čuvanje PDF fajla.
- PDF će sadržati tabelu sa podacima, grafikon i statistike.

## Format Excel Fajla

Excel fajl mora biti u .xlsx formatu. Prva kolona je obično indeks (redni broj merenja), a ostale kolone sadrže podatke.

### Primer Strukture Excel Fajla:

| Redni broj | x (cm) | Δx (cm) | y (V) | Δy (V) |
|------------|--------|---------|-------|--------|
| 1          | 10.0   | 0.1     | 5.2   | 0.05   |
| 2          | 20.0   | 0.1     | 10.4  | 0.05   |
| 3          | 30.0   | 0.1     | 15.6  | 0.05   |
| ...        | ...    | ...     | ...   | ...    |

### Pravila za Excel Fajl:

- **Kolona 1**: Redni broj (opcionalno, može biti indeks).
- **Podatkovne kolone**: Nazivi kolona trebaju biti opisni, npr. "x (cm)" za vrednost, "Δx (cm)" za grešku.
- **Tipovi podataka**: Koristite brojeve (float ili int). Tekst će biti ignorisan.
- **Prazne ćelije**: Ostavite prazno ili koristite NaN za nedostajuće vrednosti.
- **Jedinice**: U nazivima kolona navedite jedinice u zagradama, npr. "x (cm)".
- **Greške**: Ako imate greške merenja, koristite kolone sa Δ (delta), npr. "Δx (cm)".
- **Regresija**: Aplikacija će automatski povezati kolone sa greškama ako se zovu "Δ[naziv_kolone]".

### Saveti za Excel Fajl:

- Sačuvajte kao .xlsx (ne .xls).
- Koristite prvu kolonu kao indeks ako želite.
- Za linearne regresije, x i y trebaju biti numerički.
- Maksimalno 100 redova će biti prikazano u PDF-u (zbog veličine).

## Funkcionalnosti

- **Regresija**: Linearna regresija sa R² koeficijentom.
- **Grafikon**: Scatter plot sa regresijskom linijom.
- **PDF Izveštaj**: Automatski generisan izveštaj sa tabelom, grafikonom i statistikama.
- **Formatiranje**: Naučna notacija za brojeve van opsega 0.01 - 100.
- **Statistike**: Prosek, minimum, maksimum, zapis sa greškama.

## Rešavanje Problema

- Ako se aplikacija ne pokreće, proverite da li imate Python instaliran (za izvorni kod).
- Za Excel fajlove, koristite samo .xlsx format.
- Ako regresija ne radi, proverite da li su kolone numeričke.
- PDF generisanje može potrajati nekoliko sekundi.

## Kontakt

Za pitanja ili sugestije, otvorite issue na GitHub-u.
