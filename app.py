# ============================================================
#  PAST PAPER TOPIC FREQUENCY ANALYZER
#  Programming for AI (AI-2201) | DUET — BS Artificial Intelligence
#  Run: python -m streamlit run app.py
#
#  Concepts used from each lab:
#    Lab 1 — Variables, basic data types, operations
#    Lab 2 — Control flow: if/elif/else, loops
#    Lab 3 — OOP: Topic class, Subject class, Semester class
#    Lab 4 — Lists and dictionaries to store all data
#    Lab 5 — NumPy for statistics (mean, max, std deviation)
#    Lab 6 — Pandas DataFrame for table display and styling
# ============================================================

import streamlit as st   # type: ignore # web app framework
import numpy as np        # Lab 5: numerical operations
import pandas as pd       # type: ignore # Lab 6: data tables


# ─────────────────────────────────────────────────────────────
#  LAB 4: NESTED DICTIONARY
#  Structure: Semester → Subject → Topic → List of years appeared
# ─────────────────────────────────────────────────────────────

SEMESTER_DATA = {

    # ── SEMESTER 1
    "Semester 1": {
        "Introduction to Computing": {
            "Number Systems (Binary, Octal, Hex)": [2020, 2021, 2022, 2023, 2024],
            "Logic Gates & Boolean Algebra":       [2020, 2021, 2022, 2023],
            "Computer Organization":               [2021, 2022, 2023],
            "Input / Output Devices":              [2020, 2022],
            "Memory Types (RAM, ROM, Cache)":      [2021, 2023, 2024],
            "Operating System Basics":             [2020, 2022, 2024],
            "History of Computers":                [2020, 2021],
            "Software vs Hardware":                [2022, 2023],
        },
        "Calculus & Analytical Geometry": {
            "Limits & Continuity":                 [2020, 2021, 2022, 2023, 2024],
            "Differentiation":                     [2020, 2021, 2022, 2023, 2024],
            "Integration":                         [2021, 2022, 2023, 2024],
            "Partial Derivatives":                 [2022, 2023, 2024],
            "Series & Sequences":                  [2020, 2022, 2023],
            "Vectors & Geometry":                  [2021, 2023],
            "Multivariable Calculus":              [2022, 2024],
            "Applications of Derivatives":         [2020, 2021, 2023],
        },
        "English Composition": {
            "Essay Writing":                       [2020, 2021, 2022, 2023],
            "Paragraph Development":               [2021, 2022, 2024],
            "Grammar & Sentence Structure":        [2020, 2022, 2023, 2024],
            "Reading Comprehension":               [2021, 2023],
            "Report Writing":                      [2022, 2024],
            "Precis Writing":                      [2020, 2021, 2023],
        },
        "Islamic Studies / Ethics": {
            "Foundations of Islam":                [2020, 2021, 2022],
            "Islamic Economic System":             [2021, 2023],
            "Social Ethics":                       [2022, 2024],
            "Human Rights in Islam":               [2020, 2023],
        },
    },

    # ── SEMESTER 2
    "Semester 2": {
        "Programming Fundamentals": {
            "Variables & Data Types":              [2020, 2021, 2022, 2023, 2024],
            "Conditional Statements":              [2020, 2021, 2022, 2023],
            "Loops (for / while)":                 [2020, 2021, 2022, 2023, 2024],
            "Functions":                           [2021, 2022, 2023, 2024],
            "Arrays & Strings":                    [2020, 2022, 2023, 2024],
            "Pointers":                            [2021, 2023],
            "Structures":                          [2020, 2022],
            "File Handling":                       [2021, 2024],
        },
        "Linear Algebra": {
            "Matrices & Operations":               [2020, 2021, 2022, 2023, 2024],
            "Determinants":                        [2020, 2021, 2022, 2023],
            "Systems of Linear Equations":         [2021, 2022, 2023, 2024],
            "Eigenvalues & Eigenvectors":          [2020, 2022, 2023, 2024],
            "Vector Spaces":                       [2021, 2023],
            "Linear Transformations":              [2022, 2024],
            "Orthogonality":                       [2020, 2023],
            "Singular Value Decomposition":        [2023, 2024],
        },
        "Digital Logic Design": {
            "Combinational Circuits":              [2020, 2021, 2022, 2023],
            "Sequential Circuits":                 [2021, 2022, 2023, 2024],
            "Flip Flops":                          [2020, 2022, 2024],
            "Karnaugh Maps":                       [2020, 2021, 2022, 2023, 2024],
            "Multiplexers & Demultiplexers":       [2021, 2023],
            "Encoders & Decoders":                 [2020, 2022],
            "Counters & Registers":                [2021, 2023, 2024],
            "Adders & Subtractors":                [2020, 2021, 2022],
        },
        "Pakistan Studies": {
            "Ideology of Pakistan":                [2020, 2021, 2022, 2023],
            "Independence Movement":               [2020, 2022, 2023],
            "Constitution of Pakistan":            [2021, 2023, 2024],
            "Geography of Pakistan":               [2020, 2022],
            "Economic Development":                [2021, 2023],
        },
    },

    # ── SEMESTER 3
    "Semester 3": {
        "Object-Oriented Programming": {
            "Classes & Objects":                   [2020, 2021, 2022, 2023, 2024],
            "Inheritance":                         [2020, 2021, 2022, 2023, 2024],
            "Polymorphism":                        [2021, 2022, 2023, 2024],
            "Encapsulation & Abstraction":         [2020, 2022, 2023],
            "Constructors & Destructors":          [2021, 2023, 2024],
            "Operator Overloading":                [2020, 2022, 2024],
            "Exception Handling":                  [2021, 2023],
            "Templates & Generics":                [2022, 2024],
        },
        "Discrete Structures": {
            "Sets & Set Operations":               [2020, 2021, 2022, 2023],
            "Relations":                           [2021, 2022, 2024],
            "Functions":                           [2020, 2022, 2023],
            "Propositional Logic":                 [2020, 2021, 2022, 2023, 2024],
            "Predicate Logic":                     [2021, 2023, 2024],
            "Mathematical Induction":              [2021, 2022, 2023, 2024],
            "Graph Theory Basics":                 [2021, 2022, 2023, 2024],
            "Boolean Algebra":                     [2020, 2021, 2022],
            "Counting & Permutations":             [2021, 2023],
            "Trees & Spanning Trees":              [2022, 2024],
        },
        "Probability & Statistics": {
            "Basic Probability":                   [2020, 2021, 2022, 2023, 2024],
            "Conditional Probability":             [2020, 2021, 2022, 2023],
            "Bayes Theorem":                       [2021, 2022, 2023, 2024],
            "Random Variables":                    [2020, 2022, 2023],
            "Probability Distributions":           [2021, 2022, 2023, 2024],
            "Normal Distribution":                 [2020, 2022, 2024],
            "Hypothesis Testing":                  [2021, 2023],
            "Regression Analysis":                 [2022, 2024],
        },
        "Introduction to Economics": {
            "Supply & Demand":                     [2020, 2021, 2022, 2023, 2024],
            "Elasticity":                          [2021, 2022, 2023],
            "Market Structures":                   [2020, 2021, 2022, 2023, 2024],
            "Perfect Competition":                 [2021, 2022, 2023],
            "Monopoly":                            [2020, 2022, 2024],
            "GDP & National Income":               [2021, 2022, 2024],
            "Inflation":                           [2020, 2022, 2023],
            "Pakistan Economy":                    [2021, 2022, 2023, 2024],
            "Utility & Indifference Curves":       [2020, 2023],
            "Fiscal & Monetary Policy":            [2021, 2023],
        },
    },

    # ── SEMESTER 4
    "Semester 4": {
        "Data Structures & Algorithms": {
            "Arrays & Linked Lists":               [2020, 2021, 2022, 2023, 2024],
            "Stacks & Queues":                     [2020, 2021, 2022, 2023],
            "Trees (BST, AVL)":                    [2021, 2022, 2023, 2024],
            "Heaps & Priority Queues":             [2020, 2022, 2024],
            "Hashing":                             [2021, 2023, 2024],
            "Sorting Algorithms":                  [2020, 2021, 2022, 2023, 2024],
            "Searching Algorithms":                [2020, 2022, 2023],
            "Graph Algorithms":                    [2021, 2022, 2023, 2024],
            "Dynamic Programming":                 [2022, 2023, 2024],
            "Time & Space Complexity":             [2020, 2021, 2022, 2023, 2024],
        },
        "Computer Networks": {
            "OSI Model":                           [2020, 2021, 2022, 2023, 2024],
            "TCP/IP Model":                        [2021, 2022, 2023],
            "IP Addressing & Subnetting":          [2020, 2021, 2022, 2023, 2024],
            "VLSM":                                [2022, 2023, 2024],
            "Routing Algorithms":                  [2020, 2022, 2024],
            "Data Link Layer":                     [2021, 2022, 2024],
            "CRC Error Detection":                 [2020, 2023, 2024],
            "ARQ Protocols":                       [2021, 2022],
            "DNS & DHCP":                          [2021, 2023],
            "Network Security Basics":             [2021, 2024],
        },
        "Programming for AI": {
            "Variables & Data Types":              [2021, 2022, 2023],
            "Loops (for / while)":                 [2021, 2022, 2023, 2024],
            "Object-Oriented Programming":         [2020, 2021, 2022, 2023, 2024],
            "Lists & Tuples":                      [2021, 2023, 2024],
            "NumPy Arrays":                        [2022, 2023, 2024],
            "NumPy Statistical Operations":        [2023, 2024],
            "Pandas DataFrames":                   [2022, 2023, 2024],
            "Data Filtering & Sorting":            [2023, 2024],
            "Data Visualization":                  [2023, 2024],
            "Exception Handling":                  [2023],
        },
        "Software Engineering": {
            "SDLC Models":                         [2020, 2021, 2022, 2023, 2024],
            "Waterfall Model":                     [2020, 2021, 2023],
            "Agile Methodology":                   [2022, 2023, 2024],
            "Scrum Framework":                     [2023, 2024],
            "Requirements Engineering":            [2020, 2022, 2024],
            "UML Diagrams":                        [2020, 2022, 2023, 2024],
            "Software Testing":                    [2020, 2021, 2022, 2023],
            "Software Architecture":               [2022, 2024],
            "Project Management":                  [2021, 2023, 2024],
            "Version Control (Git)":               [2023, 2024],
        },
    },

    # ── SEMESTER 5
    "Semester 5": {
        "AI Fundamentals": {
            "Introduction to AI":                  [2020, 2021, 2022, 2023, 2024],
            "Intelligent Agents":                  [2021, 2022, 2024],
            "PEAS Framework":                      [2020, 2022, 2023],
            "Search Algorithms":                   [2021, 2022, 2023, 2024],
            "BFS & DFS":                           [2020, 2022, 2024],
            "A* Algorithm":                        [2021, 2023, 2024],
            "Knowledge Representation":            [2020, 2021, 2023],
            "Expert Systems":                      [2020, 2022],
            "Machine Learning Basics":             [2021, 2022, 2023, 2024],
            "Turing Test & AI Ethics":             [2020, 2021, 2023],
        },
        "Machine Learning": {
            "Supervised Learning":                 [2020, 2021, 2022, 2023, 2024],
            "Unsupervised Learning":               [2021, 2022, 2023, 2024],
            "Linear Regression":                   [2020, 2021, 2022, 2023, 2024],
            "Logistic Regression":                 [2021, 2022, 2023],
            "Decision Trees":                      [2020, 2022, 2023, 2024],
            "Support Vector Machines":             [2021, 2023, 2024],
            "K-Means Clustering":                  [2020, 2022, 2024],
            "Overfitting & Regularization":        [2021, 2022, 2023, 2024],
            "Cross Validation":                    [2022, 2023, 2024],
            "Feature Engineering":                 [2021, 2023],
        },
        "Database Systems": {
            "ER Diagrams":                         [2020, 2021, 2022, 2023, 2024],
            "Relational Model":                    [2020, 2021, 2022, 2023],
            "SQL Queries":                         [2020, 2021, 2022, 2023, 2024],
            "Normalization (1NF–3NF)":             [2021, 2022, 2023, 2024],
            "Transactions & ACID":                 [2020, 2022, 2024],
            "Indexing & Hashing":                  [2021, 2023],
            "Stored Procedures & Triggers":        [2022, 2024],
            "NoSQL Databases":                     [2023, 2024],
        },
        "Operating Systems": {
            "Process Management":                  [2020, 2021, 2022, 2023, 2024],
            "CPU Scheduling":                      [2020, 2021, 2022, 2023],
            "Memory Management":                   [2021, 2022, 2023, 2024],
            "Virtual Memory & Paging":             [2020, 2022, 2024],
            "Deadlocks":                           [2021, 2022, 2023, 2024],
            "File Systems":                        [2020, 2022, 2023],
            "Semaphores & Synchronization":        [2021, 2023, 2024],
            "I/O Management":                      [2022, 2024],
        },
    },

    # ── SEMESTER 6
    "Semester 6": {
        "Deep Learning": {
            "Neural Network Basics":               [2021, 2022, 2023, 2024],
            "Backpropagation":                     [2021, 2022, 2023, 2024],
            "Convolutional Neural Networks":       [2020, 2021, 2022, 2023, 2024],
            "Recurrent Neural Networks":           [2021, 2022, 2023],
            "LSTM & GRU":                          [2022, 2023, 2024],
            "Transfer Learning":                   [2021, 2023, 2024],
            "Activation Functions":                [2020, 2022, 2023],
            "Dropout & Batch Normalization":       [2022, 2023, 2024],
            "Optimizers (SGD, Adam)":              [2021, 2023],
            "Generative Adversarial Networks":     [2022, 2024],
        },
        "Computer Vision": {
            "Image Representation":                [2020, 2021, 2022, 2023],
            "Image Filtering & Convolution":       [2021, 2022, 2023, 2024],
            "Edge Detection":                      [2020, 2022, 2023, 2024],
            "Object Detection":                    [2021, 2022, 2023, 2024],
            "Image Segmentation":                  [2022, 2023, 2024],
            "Feature Extraction (SIFT, HOG)":      [2020, 2022, 2024],
            "Face Recognition":                    [2021, 2023],
            "OpenCV Basics":                       [2022, 2024],
        },
        "Natural Language Processing": {
            "Text Preprocessing":                  [2021, 2022, 2023, 2024],
            "Tokenization & Stemming":             [2020, 2022, 2023],
            "Bag of Words & TF-IDF":               [2021, 2022, 2023, 2024],
            "Word Embeddings (Word2Vec)":           [2022, 2023, 2024],
            "Sentiment Analysis":                  [2021, 2023, 2024],
            "Named Entity Recognition":            [2022, 2023],
            "Language Models":                     [2021, 2022, 2023, 2024],
            "Transformers & Attention":            [2022, 2023, 2024],
        },
        "Theory of Automata": {
            "Finite Automata (DFA/NFA)":           [2020, 2021, 2022, 2023, 2024],
            "Regular Expressions":                 [2020, 2021, 2022, 2023],
            "Context-Free Grammars":               [2021, 2022, 2023, 2024],
            "Pushdown Automata":                   [2020, 2022, 2023],
            "Turing Machines":                     [2021, 2022, 2023, 2024],
            "Pumping Lemma":                       [2020, 2022, 2024],
            "Decidability":                        [2021, 2023],
            "Complexity Classes (P, NP)":          [2022, 2023, 2024],
        },
    },

    # ── SEMESTER 7
    "Semester 7": {
        "Reinforcement Learning": {
            "Markov Decision Processes":           [2021, 2022, 2023, 2024],
            "Q-Learning":                          [2021, 2022, 2023, 2024],
            "Policy Gradient Methods":             [2022, 2023, 2024],
            "Exploration vs Exploitation":         [2020, 2022, 2023],
            "Deep Q-Network (DQN)":                [2022, 2023, 2024],
            "Actor-Critic Methods":                [2022, 2024],
            "Reward Functions":                    [2021, 2023],
            "Multi-Agent RL":                      [2023, 2024],
        },
        "Big Data Analytics": {
            "Hadoop & MapReduce":                  [2021, 2022, 2023],
            "Apache Spark":                        [2022, 2023, 2024],
            "Data Warehousing":                    [2020, 2021, 2022, 2023],
            "ETL Processes":                       [2021, 2023],
            "Data Mining Techniques":              [2020, 2022, 2023, 2024],
            "Stream Processing":                   [2022, 2024],
            "NoSQL at Scale":                      [2021, 2023],
            "Visualization of Big Data":           [2022, 2024],
        },
        "Cloud Computing": {
            "Cloud Service Models (IaaS/PaaS/SaaS)": [2021, 2022, 2023, 2024],
            "Virtualization":                      [2020, 2021, 2022, 2023],
            "Docker & Containers":                 [2022, 2023, 2024],
            "Kubernetes":                          [2023, 2024],
            "AWS / Azure Basics":                  [2022, 2023, 2024],
            "Serverless Computing":                [2023, 2024],
            "Cloud Security":                      [2021, 2023],
            "Load Balancing & Scaling":            [2022, 2024],
        },
        "AI Ethics & Society": {
            "Bias in AI":                          [2021, 2022, 2023, 2024],
            "Fairness & Accountability":           [2022, 2023, 2024],
            "Privacy & Surveillance":              [2021, 2023, 2024],
            "Explainable AI (XAI)":                [2022, 2023, 2024],
            "AI Regulation & Policy":              [2021, 2023],
            "AI & Employment":                     [2022, 2024],
            "Autonomous Systems Ethics":           [2021, 2023, 2024],
            "Responsible AI Development":          [2022, 2024],
        },
    },

    # ── SEMESTER 8
    "Semester 8": {
        "Final Year Project": {
            "Project Proposal Writing":            [2020, 2021, 2022, 2023, 2024],
            "Literature Review":                   [2020, 2021, 2022, 2023, 2024],
            "Methodology Design":                  [2021, 2022, 2023, 2024],
            "Data Collection & Preprocessing":     [2022, 2023, 2024],
            "Model Development":                   [2021, 2022, 2023, 2024],
            "Results & Evaluation":                [2020, 2022, 2023, 2024],
            "Report Writing":                      [2020, 2021, 2022, 2023, 2024],
            "Project Presentation":                [2020, 2021, 2022, 2023, 2024],
        },
        "Advanced Machine Learning": {
            "Ensemble Methods":                    [2021, 2022, 2023, 2024],
            "Random Forests":                      [2021, 2022, 2023],
            "Gradient Boosting (XGBoost)":         [2022, 2023, 2024],
            "Dimensionality Reduction (PCA)":      [2020, 2021, 2022, 2023, 2024],
            "Anomaly Detection":                   [2022, 2023, 2024],
            "Time Series Analysis":                [2021, 2023, 2024],
            "AutoML":                              [2023, 2024],
            "Model Deployment":                    [2022, 2023, 2024],
        },
        "Information Security": {
            "Cryptography Basics":                 [2020, 2021, 2022, 2023, 2024],
            "Symmetric Encryption (AES, DES)":     [2020, 2021, 2022, 2023],
            "Asymmetric Encryption (RSA)":         [2021, 2022, 2023, 2024],
            "Hashing & Digital Signatures":        [2020, 2022, 2023, 2024],
            "Network Security Protocols":          [2021, 2022, 2023],
            "Firewalls & IDS":                     [2020, 2022, 2024],
            "Penetration Testing":                 [2021, 2023, 2024],
            "AI in Cybersecurity":                 [2022, 2023, 2024],
        },
        "Entrepreneurship & Management": {
            "Business Plan Development":           [2020, 2021, 2022, 2023],
            "Tech Startups":                       [2021, 2022, 2023, 2024],
            "Marketing Fundamentals":              [2020, 2022, 2023],
            "Financial Literacy":                  [2021, 2023, 2024],
            "Project Management (Agile)":          [2022, 2023, 2024],
            "Leadership & Teamwork":               [2020, 2021, 2022],
            "Intellectual Property":               [2021, 2023],
            "Innovation & Design Thinking":        [2022, 2024],
        },
    },
}


# ─────────────────────────────────────────────────────────────
#  LAB 3: OOP — Topic, Subject, Semester classes
#  Each class encapsulates its own data and logic cleanly
# ─────────────────────────────────────────────────────────────

class Topic:
    """
    Represents a single exam topic.
    Lab 3: OOP — stores name and the years it appeared.
    Lab 2: Control flow used in priority() to classify importance.
    """
    def __init__(self, name: str, years: list):
        self.name  = name
        # Lab 4: list of integers (years)
        self.years = sorted(years)

    @property
    def frequency(self) -> int:
        # Lab 1: simple variable — count of appearances
        return len(self.years)

    def priority(self, max_freq: int) -> str:
        """
        Lab 2: if/elif/else control flow.
        Compares this topic's frequency against the subject maximum
        to decide if it's High, Medium, or Low priority.
        """
        if max_freq == 0:
            return "Low"
        ratio = self.frequency / max_freq   # Lab 1: arithmetic operation
        if ratio >= 0.6:
            return "High"
        elif ratio >= 0.3:
            return "Medium"
        else:
            return "Low"


class Subject:
    """
    Represents a university subject containing multiple Topic objects.
    Lab 3: OOP — aggregates Topic objects and exposes stats.
    Lab 5: NumPy used in numpy_stats() for numerical analysis.
    Lab 6: Pandas used in to_dataframe() for table display.
    """
    def __init__(self, name: str, topic_data: dict):
        self.name = name
        # Lab 4: build a list of Topic objects from a dictionary
        self.topics = [Topic(t, y) for t, y in topic_data.items()]

    def get_topic(self, name: str):
        """Lab 2: loop to search for a topic by name."""
        for t in self.topics:
            if t.name == name:
                return t
        return None

    def max_frequency(self) -> int:
        """Lab 2: returns the highest frequency among all topics."""
        return max((t.frequency for t in self.topics), default=0)

    def to_dataframe(self) -> pd.DataFrame:
        """
        Lab 6: Pandas — converts topic list into a DataFrame
        sorted by frequency descending.
        """
        max_f = self.max_frequency()
        rows  = []
        # Lab 2: loop through topics, Lab 4: build list of dicts
        for t in sorted(self.topics, key=lambda x: -x.frequency):
            rows.append({
                "Topic":     t.name,
                "Appeared":  t.frequency,
                "Last Seen": t.years[-1] if t.years else "—",
                "Years":     ", ".join(str(y) for y in t.years),
                "Priority":  t.priority(max_f),
            })
        return pd.DataFrame(rows)   # Lab 6: Pandas DataFrame

    def numpy_stats(self) -> dict:
        """
        Lab 5: NumPy — computes mean, max, and std deviation
        of topic frequencies across this subject.
        """
        # Lab 4: list comprehension → NumPy array
        freqs = np.array([t.frequency for t in self.topics])
        return {
            "mean": round(float(np.mean(freqs)), 1),   # average frequency
            "max":  int(np.max(freqs)),                 # most frequent topic
            "std":  round(float(np.std(freqs)), 1),     # spread in frequencies
        }


class Semester:
    """
    Represents one semester containing multiple Subject objects.
    Lab 3: OOP — top-level container in the hierarchy.
    """
    def __init__(self, name: str, subject_data: dict):
        self.name     = name
        # Lab 4: dictionary of Subject objects keyed by subject name
        self.subjects = {s: Subject(s, t) for s, t in subject_data.items()}

    def get_subject(self, name: str) -> Subject:
        return self.subjects.get(name)

    def subject_names(self) -> list:
        # Lab 4: return keys of the dictionary as a list
        return list(self.subjects.keys())


# ─────────────────────────────────────────────────────────────
#  BUILD ALL SEMESTER OBJECTS  (Lab 4: dictionary of Semester objects)
# ─────────────────────────────────────────────────────────────

ALL_SEMESTERS = {
    name: Semester(name, subjects)
    for name, subjects in SEMESTER_DATA.items()
}


# ─────────────────────────────────────────────────────────────
#  PAGE CONFIG
# ─────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="Past Paper Analyzer",
    page_icon="📄",
    layout="centered"
)

# ─────────────────────────────────────────────────────────────
#  CUSTOM CSS — dark editorial design
# ─────────────────────────────────────────────────────────────

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@600;700;800&family=DM+Sans:wght@300;400;500&display=swap');

html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
    background-color: #0C0C0E;
    color: #F0EDE8;
}
.stApp { background-color: #0C0C0E; }

h1, h2, h3, h4 {
    font-family: 'Syne', sans-serif !important;
    color: #F0EDE8 !important;
    letter-spacing: -0.5px;
}

/* selectbox styling */
[data-testid="stSelectbox"] > div > div {
    background: #17171A !important;
    border: 1px solid #2C2C32 !important;
    border-radius: 10px !important;
    color: #F0EDE8 !important;
    font-size: 15px !important;
}
label {
    color: #888 !important;
    font-size: 12px !important;
    letter-spacing: 0.8px;
    text-transform: uppercase;
}

/* result card */
.result-card {
    background: linear-gradient(135deg, #17171A 0%, #1C1C22 100%);
    border: 1px solid #2C2C32;
    border-radius: 16px;
    padding: 32px 36px;
    margin: 20px 0;
    position: relative;
    overflow: hidden;
}
/* colored top bar per priority */
.result-card::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 3px;
}
.card-high::before   { background: linear-gradient(90deg, #FF6B6B, #FF8E53); }
.card-medium::before { background: linear-gradient(90deg, #FFD93D, #FF9A3C); }
.card-low::before    { background: linear-gradient(90deg, #6BCB77, #4D9FEC); }

.topic-title {
    font-family: 'Syne', sans-serif;
    font-size: 1.5rem;
    font-weight: 700;
    color: #F0EDE8;
    margin: 0 0 4px 0;
}
.subject-tag {
    font-size: 12px;
    color: #555;
    text-transform: uppercase;
    letter-spacing: 1px;
    margin-bottom: 22px;
}

/* priority badge */
.badge {
    display: inline-block;
    padding: 5px 16px;
    border-radius: 30px;
    font-size: 12px;
    font-weight: 700;
    letter-spacing: 1px;
    text-transform: uppercase;
    margin-bottom: 22px;
}
.badge-high   { background: rgba(255,107,107,0.15); color: #FF6B6B; border: 1px solid rgba(255,107,107,0.3); }
.badge-medium { background: rgba(255,217,61,0.12);  color: #FFD93D; border: 1px solid rgba(255,217,61,0.25); }
.badge-low    { background: rgba(107,203,119,0.12); color: #6BCB77; border: 1px solid rgba(107,203,119,0.25); }

/* big frequency number */
.freq-number {
    font-family: 'Syne', sans-serif;
    font-size: 4rem;
    font-weight: 800;
    line-height: 1;
    margin-bottom: 2px;
}
.freq-high   { color: #FF6B6B; }
.freq-medium { color: #FFD93D; }
.freq-low    { color: #6BCB77; }

.freq-label {
    font-size: 12px;
    color: #444;
    text-transform: uppercase;
    letter-spacing: 1px;
    margin-bottom: 26px;
}

/* year pills */
.years-label {
    font-size: 11px;
    color: #555;
    text-transform: uppercase;
    letter-spacing: 1px;
    margin-bottom: 10px;
}
.year-pill {
    display: inline-block;
    background: #222228;
    border: 1px solid #333340;
    color: #CCC;
    border-radius: 8px;
    padding: 6px 16px;
    margin: 3px;
    font-size: 14px;
    font-weight: 500;
    font-family: 'Syne', sans-serif;
}

/* verdict box */
.verdict {
    margin-top: 22px;
    padding: 14px 18px;
    background: rgba(255,255,255,0.03);
    border-radius: 10px;
    border-left: 3px solid #2C2C32;
    font-size: 14px;
    color: #AAA;
    line-height: 1.6;
}

/* stat boxes */
.stat-box {
    background: #17171A;
    border: 1px solid #2C2C32;
    border-radius: 12px;
    padding: 18px 20px;
    text-align: center;
}
.stat-val {
    font-family: 'Syne', sans-serif;
    font-size: 1.8rem;
    font-weight: 700;
    color: #F0EDE8;
}
.stat-lbl {
    font-size: 11px;
    color: #555;
    text-transform: uppercase;
    letter-spacing: 1px;
    margin-top: 4px;
}

/* section titles */
.section-title {
    font-family: 'Syne', sans-serif;
    font-size: 12px;
    text-transform: uppercase;
    letter-spacing: 2px;
    color: #444;
    margin: 28px 0 10px 0;
}

/* table */
[data-testid="stDataFrame"] {
    border: 1px solid #2C2C32 !important;
    border-radius: 12px !important;
    overflow: hidden !important;
}

hr { border-color: #1E1E24 !important; }
#MainMenu, footer { visibility: hidden; }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────
#  HEADER
# ─────────────────────────────────────────────────────────────

st.markdown("""
<div style="padding: 8px 0 24px 0;">
    <div style="font-size:11px; text-transform:uppercase; letter-spacing:3px; color:#444; margin-bottom:8px;">
        DUET · BS Artificial Intelligence · AI-2201
    </div>
    <h1 style="margin:0; font-size:2rem; font-family:'Syne',sans-serif;">
        Past Paper Analyzer
    </h1>
    <p style="color:#555; margin-top:6px; font-size:14px;">
        Select your semester, subject, and topic — see its full exam history instantly.
    </p>
</div>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────
#  STEP 1 — SEMESTER DROPDOWN
# ─────────────────────────────────────────────────────────────

# Lab 4: list of semester names from dictionary keys
semester_names  = list(ALL_SEMESTERS.keys())
chosen_semester = st.selectbox("Semester", semester_names)

semester_obj    = ALL_SEMESTERS[chosen_semester]   # Lab 3: Semester object


# ─────────────────────────────────────────────────────────────
#  STEP 2 — SUBJECT DROPDOWN  (updates when semester changes)
# ─────────────────────────────────────────────────────────────

col1, col2 = st.columns(2)

with col1:
    # Lab 4: list of subject names for the chosen semester
    subject_names  = semester_obj.subject_names()
    chosen_subject = st.selectbox("Subject", subject_names)

subject_obj = semester_obj.get_subject(chosen_subject)   # Lab 3: Subject object


# ─────────────────────────────────────────────────────────────
#  STEP 3 — TOPIC DROPDOWN  (updates when subject changes)
# ─────────────────────────────────────────────────────────────

with col2:
    # Lab 4: list of topic names from the Subject's topic list
    topic_names  = [t.name for t in subject_obj.topics]
    chosen_topic = st.selectbox("Topic", topic_names)

topic_obj = subject_obj.get_topic(chosen_topic)   # Lab 3: Topic object
max_f     = subject_obj.max_frequency()


# ─────────────────────────────────────────────────────────────
#  RESULT CARD — shows years + priority for selected topic
# ─────────────────────────────────────────────────────────────

if topic_obj:
    # Lab 2: control flow to pick styling based on priority
    prio = topic_obj.priority(max_f)

    card_class  = {"High": "card-high",   "Medium": "card-medium",   "Low": "card-low"}[prio]
    badge_class = {"High": "badge-high",  "Medium": "badge-medium",  "Low": "badge-low"}[prio]
    freq_class  = {"High": "freq-high",   "Medium": "freq-medium",   "Low": "freq-low"}[prio]
    badge_text  = {"High": "🔴 High Priority", "Medium": "🟡 Medium Priority", "Low": "🟢 Low Priority"}[prio]

    # Lab 2: if/elif/else for verdict text
    if prio == "High":
        verdict = "This topic shows up constantly. If you study nothing else, study this."
    elif prio == "Medium":
        verdict = "Appeared a few times — solid chance it comes back. Worth revising."
    else:
        verdict = "Rarely seen. Cover it only if you've finished everything else."

    # Lab 4: list → joined string for year pills
    years_html = "".join(
        f'<span class="year-pill">{y}</span>' for y in topic_obj.years
    )

    st.markdown(f"""
    <div class="result-card {card_class}">
        <div class="topic-title">{topic_obj.name}</div>
        <div class="subject-tag">{chosen_subject} &nbsp;·&nbsp; {chosen_semester}</div>
        <div><span class="badge {badge_class}">{badge_text}</span></div>
        <div class="freq-number {freq_class}">{topic_obj.frequency}<span style="font-size:1.5rem; color:#333;">×</span></div>
        <div class="freq-label">times in past papers</div>
        <div class="years-label">appeared in</div>
        <div>{years_html}</div>
        <div class="verdict">💡 {verdict}</div>
    </div>
    """, unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────
#  STATS ROW — NumPy stats for the selected subject
# ─────────────────────────────────────────────────────────────

# Lab 5: NumPy statistics via Subject.numpy_stats()
stats = subject_obj.numpy_stats()

st.markdown('<div class="section-title">Subject Statistics</div>', unsafe_allow_html=True)
c1, c2, c3 = st.columns(3)
with c1:
    st.markdown(f'<div class="stat-box"><div class="stat-val">{stats["max"]}×</div><div class="stat-lbl">Most Frequent</div></div>', unsafe_allow_html=True)
with c2:
    st.markdown(f'<div class="stat-box"><div class="stat-val">{stats["mean"]}</div><div class="stat-lbl">Avg Appearances</div></div>', unsafe_allow_html=True)
with c3:
    st.markdown(f'<div class="stat-box"><div class="stat-val">{stats["std"]}</div><div class="stat-lbl">Std Deviation</div></div>', unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────
#  FULL TABLE — all topics ranked for selected subject
# ─────────────────────────────────────────────────────────────

st.markdown('<div class="section-title">All Topics Ranked</div>', unsafe_allow_html=True)

# Lab 6: Pandas DataFrame from Subject.to_dataframe()
df = subject_obj.to_dataframe()

# Lab 6: Pandas Styler — color the Priority column
# Note: use .map() not .applymap() for newer Pandas versions
def color_priority(val):
    """Lab 2: control flow to return CSS color string per priority value."""
    if val == "High":   return "background-color:#2A1515; color:#FF6B6B;"
    if val == "Medium": return "background-color:#252010; color:#FFD93D;"
    return "background-color:#0F1F12; color:#6BCB77;"

st.dataframe(
    df.style.map(color_priority, subset=["Priority"]),  # Lab 6: Pandas styling
    use_container_width=True,
    hide_index=True,
    height=320,
)