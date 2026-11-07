# Forecast comparison: 2026 week 9

Generated 2026-11-05 16:16. Games in kickoff order. Each model has a winner pick with its win probability, then its ATS pick: 1 point(s) of predicted margin per 5 percentage points of confidence above 50%, against the line in `data/games.csv`. A blank ATS pick means the predicted margin equals the line.

- Power rankings: local Bayesian agent, features `rank_diff`.
- Nick's tiers: local Bayesian agent, features `nw_rank_diff`.
- Combined: local Bayesian agent, features `rank_diff+nw_rank_diff`.
- LLM blind: `trace_2026_w09_llm-blind_t1.2.jsonl` (Qwen3.8-Flash-Next, temperature 1.2).
- LLM informed: `trace_2026_w09_llm-informed_t1.2.jsonl` (Qwen3.8-Flash-Next, temperature 1.2).

| Game                | Line      | Power rankings   | Power rankings ATS   | Nick's tiers   | Nick's tiers ATS   | Combined   | Combined ATS   | LLM blind   | LLM blind ATS   | LLM informed   | LLM informed ATS   |
|:--------------------|:----------|:-----------------|:---------------------|:---------------|:-------------------|:-----------|:---------------|:------------|:----------------|:---------------|:-------------------|
| JAX @ BAL           | JAX -9.5  | JAX 0.70         | BAL                  | JAX 0.73       | BAL                | JAX 0.73   | BAL            | BAL 0.67    | BAL             | JAX 0.66       | BAL                |
| CIN @ ATL (neutral) | CIN -3    | CIN 0.58         | ATL                  | CIN 0.62       | ATL                | CIN 0.61   | ATL            | ATL 0.55    | ATL             | CIN 0.55       | ATL                |
| DEN @ CAR           | DEN -2.5  | DEN 0.66         | DEN                  | DEN 0.66       | DEN                | DEN 0.69   | DEN            | DEN 0.56    | CAR             | DEN 0.64       | DEN                |
| DAL @ IND           | DAL -6    | DAL 0.56         | IND                  | DAL 0.65       | IND                | DAL 0.59   | IND            | IND 0.62    | IND             | DAL 0.63       | IND                |
| NYJ @ KC            | KC -9.5   | KC 0.77          | NYJ                  | KC 0.83        | NYJ                | KC 0.80    | NYJ            | KC 0.84     | NYJ             | KC 0.82        | NYJ                |
| DET @ MIA           | DET -7    | DET 0.74         | MIA                  | DET 0.75       | MIA                | DET 0.77   | MIA            | DET 0.58    | MIA             | DET 0.83       | MIA                |
| CLE @ NO            | NO -2.5   | CLE 0.64         | CLE                  | CLE 0.52       | CLE                | CLE 0.65   | CLE            | CLE 0.65    | CLE             | CLE 0.62       | CLE                |
| NYG @ PHI           | NYG -3.5  | NYG 0.70         | NYG                  | NYG 0.62       | PHI                | NYG 0.72   | NYG            | PHI 0.78    | PHI             | NYG 0.66       | PHI                |
| LA @ WAS            | LA -5.5   | LA 0.57          | WAS                  | LA 0.65        | WAS                | LA 0.60    | WAS            | WAS 0.58    | WAS             | LA 0.55        | WAS                |
| HOU @ LAC           | HOU -4.5  | HOU 0.61         | LAC                  | HOU 0.52       | LAC                | HOU 0.62   | LAC            | LAC 0.57    | LAC             | HOU 0.72       | LAC                |
| LV @ SF             | SF -10    | SF 0.76          | LV                   | SF 0.74        | LV                 | SF 0.79    | LV             | SF 0.72     | LV              | SF 0.74        | LV                 |
| GB @ NE             | NE -2.5   | NE 0.57          | GB                   | NE 0.54        | GB                 | NE 0.57    | GB             | NE 0.57     | GB              | NE 0.56        | GB                 |
| ARI @ SEA           | SEA -10.5 | SEA 0.74         | ARI                  | SEA 0.77       | ARI                | SEA 0.77   | ARI            | SEA 0.75    | ARI             | SEA 0.62       | ARI                |
| TB @ CHI            | CHI -6    | CHI 0.71         | TB                   | CHI 0.77       | TB                 | CHI 0.74   | TB             | CHI 0.58    | TB              | CHI 0.68       | TB                 |
| BUF @ MIN           | BUF -3.5  | BUF 0.57         | MIN                  | BUF 0.61       | MIN                | BUF 0.60   | MIN            | BUF 0.58    | MIN             | BUF 0.58       | MIN                |
