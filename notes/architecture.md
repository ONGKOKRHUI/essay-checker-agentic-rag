```mermaid
flowchart TD
    A[input: student essay content] -->|pdf| B(PDF loader)
    A --> |image| C[computer vision]
    C --> D{text}
    B --> D
    J[fact extractor - implemented]
    D --> |string| J
    subgraph K[Agentic RAG fact checker]
        subgraph O[Tools]
            L[web search - crawler]
            M[knowledge base - PGVector]
        end
        subgraph P[instructions - ordered prompt - for each fact]
            Q[1. generic question: no tool - optional]
            R[2. use knowledge base tool - single iteration]
            S[3. use web search tool - k iterations]
        end
    end

    subgraph G[knowledge base - data]
        H[essay_question]
        I[essay_rubric]
    end
    J --> |JSON| K
    subgraph T[LLM logic checker]
        U[prompt engineering]
        V[OTHERS KIV]
    end
    D --> |data preprocessing for parsing and overlap paragraph| T
    W[LLM to judge the language and grammar]
    D --> |data preprocessing if parsing needed| W

    G --> |raw text in pdf| X[LLM for question and rubrics extraction and organization]
    subgraph Z[Final LLM as a judge]
        AA[matches inputs and compare with rubrics JSON]
        AB[generates a final detailed report]
    end
    
    T --> |provide a logic score| Z
    W --> |provide language and grammar score| Z
    K --> |provide a fact reliability score| Z
    X --> |Rubrics in JSON| Z