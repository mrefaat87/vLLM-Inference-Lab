"""
Stage 2 — Experiment 1: Reach the Breaking Point — Preemption and Queuing

With prefix caching DISABLED and 8-bit GPTQ model:
- KV cache: 4.21 GiB = 78,848 tokens
- Each request: ~500 input + 1500 output = ~2000 tokens KV
- Max concurrent: 78,848 / 2000 ≈ 39
- Test N = 10, 20, 30, 35, 38, 40, 45, 50 to find the breaking point

Usage: python3 stage2_exp1_preemption.py --host http://<ip>:8000
       [--ssh-key ~/.ssh/vllm-lab-key-2.pem] [--ssh-host ubuntu@44.201.96.75]
"""

import json
import time
import threading
import argparse
import requests
import subprocess
import sys
import statistics

# --- Configuration ---
HOST = ""
MODEL = "Qwen/Qwen2.5-7B-Instruct-GPTQ-Int8"
MAX_TOKENS = 1500
SSH_KEY = ""
SSH_HOST = ""

CONCURRENCY_LEVELS = [10, 20, 30, 35, 38, 40, 45, 50]

# --- 40 Unique Prompts (different topics, phrasing, ~400-500 words each) ---
ESSAY_TOPICS = [
    ("the history of the Roman Empire",
     "Cover its founding myths, the Republic era, the transition to Empire under Augustus, "
     "the Pax Romana, major emperors like Trajan and Hadrian, the crisis of the third century, "
     "the split into Eastern and Western halves, and the eventual fall of the Western Roman Empire in 476 AD. "
     "Discuss the social structure including patricians and plebeians, the role of the Roman Senate, "
     "military innovations like the legion system, engineering achievements such as aqueducts and roads, "
     "and the lasting cultural legacy on modern Western civilization including law, language, and architecture."),

    ("quantum computing and its potential impact on technology",
     "Explain the fundamental principles of quantum mechanics that enable quantum computing, "
     "including superposition, entanglement, and quantum interference. Describe how qubits differ "
     "from classical bits, the challenges of quantum decoherence, and current approaches to building "
     "quantum computers such as superconducting circuits, trapped ions, and topological qubits. "
     "Discuss potential applications in cryptography, drug discovery, optimization problems, and "
     "machine learning. Address the current limitations and the timeline for achieving quantum advantage."),

    ("marine biology and the ecosystems of the deep ocean",
     "Explore the vast diversity of life in the world's oceans, from coral reefs teeming with colorful "
     "fish to the mysterious creatures of the abyssal zone. Discuss the importance of phytoplankton as "
     "the base of the marine food web, the ecology of hydrothermal vents, bioluminescence in deep-sea "
     "organisms, and the migratory patterns of whales and sea turtles. Address threats like ocean "
     "acidification, overfishing, plastic pollution, and coral bleaching due to rising temperatures."),

    ("the Renaissance period in European art and culture",
     "Trace the origins of the Renaissance in 14th-century Florence and its spread across Europe. "
     "Discuss key figures like Leonardo da Vinci, Michelangelo, Raphael, and Botticelli and their "
     "revolutionary contributions to painting, sculpture, and architecture. Examine the role of patrons "
     "like the Medici family, the invention of linear perspective, the revival of classical Greek and "
     "Roman ideals, and how the printing press accelerated the spread of Renaissance ideas. "
     "Compare the Italian Renaissance with the Northern Renaissance including artists like Albrecht Durer."),

    ("the science of climate change and global warming",
     "Present the scientific evidence for human-caused climate change, including the greenhouse effect, "
     "rising CO2 levels from fossil fuel combustion, deforestation contributions, and methane emissions "
     "from agriculture. Discuss observed impacts like retreating glaciers, rising sea levels, more intense "
     "hurricanes, shifting precipitation patterns, and threats to biodiversity. Examine potential solutions "
     "including renewable energy transition, carbon capture technology, international agreements like the "
     "Paris Accord, and individual actions that can reduce carbon footprints."),

    ("the evolution of artificial intelligence from Turing to transformers",
     "Trace the development of AI from Alan Turing's foundational work and the Dartmouth conference "
     "through expert systems, the AI winters, the rise of machine learning, deep learning breakthroughs "
     "with convolutional and recurrent neural networks, and the transformer architecture revolution. "
     "Discuss milestone systems like Deep Blue, Watson, AlphaGo, GPT models, and their societal impacts. "
     "Address ethical concerns including bias, job displacement, autonomous weapons, and the alignment problem."),

    ("the history and cultural significance of jazz music",
     "Explore the origins of jazz in the African American communities of New Orleans in the late 19th "
     "and early 20th centuries. Discuss the influence of blues, ragtime, and brass band traditions. "
     "Cover major eras including the swing era, bebop revolution with Charlie Parker and Dizzy Gillespie, "
     "cool jazz, hard bop, free jazz with Ornette Coleman and John Coltrane, and jazz fusion. "
     "Examine how jazz influenced civil rights movements and became America's classical music."),

    ("the human immune system and how vaccines work",
     "Describe the innate and adaptive immune systems, including the roles of white blood cells, "
     "antibodies, T-cells, B-cells, and memory cells. Explain how pathogens like bacteria and viruses "
     "attack the body and how the immune response counters them. Detail the history of vaccination from "
     "Edward Jenner's smallpox vaccine to modern mRNA vaccines. Discuss herd immunity, vaccine hesitancy, "
     "autoimmune disorders, and the future of immunotherapy in treating cancer."),

    ("the philosophy of existentialism and its major thinkers",
     "Examine the core tenets of existentialism including the primacy of individual existence, freedom, "
     "authenticity, and the confrontation with absurdity. Discuss the works of Soren Kierkegaard as a "
     "precursor, then explore Jean-Paul Sartre's concept of radical freedom, Simone de Beauvoir's "
     "feminist existentialism, Albert Camus and the absurd, Martin Heidegger's being-in-the-world, "
     "and how existentialist themes appear in literature, theater, and everyday decision-making."),

    ("the architecture and engineering of ancient Egyptian pyramids",
     "Investigate how the ancient Egyptians built the Great Pyramid of Giza and other monumental structures "
     "without modern technology. Discuss theories about construction methods including ramps, levers, and "
     "water lubrication. Examine the astronomical alignments, the purpose of pyramids as royal tombs, "
     "the evolution from mastabas to step pyramids to true pyramids, the role of the workforce, "
     "and recent archaeological discoveries that have changed our understanding of pyramid construction."),

    ("the psychology of decision-making and cognitive biases",
     "Explore how humans make decisions and why they often deviate from rationality. Discuss Daniel "
     "Kahneman's dual-process theory of System 1 and System 2 thinking. Cover major cognitive biases "
     "including confirmation bias, anchoring effect, availability heuristic, sunk cost fallacy, "
     "Dunning-Kruger effect, and framing effects. Examine applications in behavioral economics, "
     "marketing, public policy nudges, and strategies for improving personal decision-making."),

    ("the exploration and colonization of Mars",
     "Assess the current state of Mars exploration from rovers like Curiosity and Perseverance to plans "
     "by NASA and SpaceX for human missions. Discuss the technical challenges of getting to Mars including "
     "propulsion, radiation shielding, life support systems, and landing on the Martian surface. "
     "Examine the feasibility of terraforming, establishing permanent settlements, growing food on Mars, "
     "the legal framework under the Outer Space Treaty, and the ethical questions of planetary colonization."),

    ("the industrial revolution and its transformation of society",
     "Analyze how the Industrial Revolution beginning in 18th-century Britain fundamentally changed "
     "human civilization. Discuss the shift from agrarian economies to factory-based manufacturing, "
     "key inventions like the steam engine, spinning jenny, and power loom. Examine the social consequences "
     "including urbanization, child labor, the rise of the middle class, labor movements, and environmental "
     "degradation. Trace the spread to America, Europe, and eventually Asia, and compare with modern "
     "digital transformation."),

    ("the biodiversity of tropical rainforests and conservation efforts",
     "Describe the incredible species richness of tropical rainforests, which contain over half of the "
     "world's species on just 6 percent of land surface. Discuss the layered structure from emergent "
     "canopy to forest floor, symbiotic relationships, medicinal plants, and indigenous communities. "
     "Examine drivers of deforestation including agriculture, logging, and mining. Evaluate conservation "
     "strategies such as protected areas, sustainable forestry, REDD+ programs, and ecotourism."),

    ("the mathematical foundations of cryptography",
     "Explain how mathematics underpins modern cryptographic systems that secure digital communications. "
     "Cover symmetric encryption like AES, asymmetric encryption using RSA and elliptic curves, "
     "hash functions, digital signatures, and zero-knowledge proofs. Discuss the number theory behind RSA "
     "including prime factorization, modular arithmetic, and Euler's totient function. "
     "Address the threat of quantum computing to current encryption and post-quantum cryptography research."),

    ("the sociology of social media and its effects on mental health",
     "Investigate how platforms like Facebook, Instagram, TikTok, and Twitter have reshaped human social "
     "interaction, self-image, and community formation. Discuss research findings on social media's "
     "correlation with anxiety, depression, loneliness, and body image issues, especially among teenagers. "
     "Examine the dopamine feedback loops designed into these platforms, the spread of misinformation, "
     "online harassment, filter bubbles, and potential regulatory approaches to mitigate harms."),

    ("the physics of black holes and general relativity",
     "Describe how Einstein's general theory of relativity predicts the existence of black holes as "
     "regions where spacetime curvature becomes extreme. Discuss the Schwarzschild radius, event horizons, "
     "singularities, and the information paradox. Cover observational evidence including the first image "
     "by the Event Horizon Telescope, gravitational wave detection by LIGO, and the discovery of "
     "supermassive black holes at galaxy centers. Address Hawking radiation and its implications."),

    ("the culinary traditions and food culture of Japan",
     "Explore the rich food heritage of Japan from ancient rice cultivation to the refinement of washoku "
     "cuisine, recognized by UNESCO as an Intangible Cultural Heritage. Discuss key elements including "
     "umami as the fifth taste, the art of sushi and sashimi preparation, ramen regional varieties, "
     "kaiseki multi-course dining, the tea ceremony, fermented foods like miso and natto, "
     "seasonal ingredients, and the meticulous presentation aesthetics that define Japanese food culture."),

    ("the history of space exploration from Sputnik to the ISS",
     "Chronicle the space race between the United States and Soviet Union, from Sputnik's launch in 1957 "
     "through Gagarin's first human spaceflight, the Mercury and Gemini programs, the Apollo moon landings, "
     "Skylab, the Space Shuttle era, and the construction of the International Space Station. "
     "Discuss the shift toward international cooperation, commercial spaceflight with SpaceX and Blue Origin, "
     "and plans for returning to the Moon through the Artemis program."),

    ("the neuroscience of memory formation and recall",
     "Explain how the brain encodes, stores, and retrieves memories at the molecular and systems level. "
     "Discuss the roles of the hippocampus, amygdala, and prefrontal cortex. Cover different types of "
     "memory including episodic, semantic, procedural, and working memory. Examine long-term potentiation, "
     "synaptic plasticity, memory consolidation during sleep, false memories, and neurodegenerative "
     "diseases like Alzheimer's that devastate memory function."),

    ("the economics of cryptocurrency and decentralized finance",
     "Analyze the economic principles behind Bitcoin, Ethereum, and the broader cryptocurrency ecosystem. "
     "Discuss blockchain technology, proof-of-work versus proof-of-stake consensus mechanisms, smart "
     "contracts, decentralized applications, yield farming, and NFTs. Examine macroeconomic implications "
     "including monetary policy challenges, financial inclusion, energy consumption concerns, regulatory "
     "approaches worldwide, and whether crypto represents a paradigm shift or speculative bubble."),

    ("the evolution of the English language over a thousand years",
     "Trace the development of English from Old English spoken by Anglo-Saxon settlers through Middle "
     "English influenced by the Norman Conquest, Early Modern English of Shakespeare's era, to present-day "
     "Global English. Discuss the Great Vowel Shift, the influence of Latin, French, Norse, and other "
     "languages on English vocabulary, the standardization through dictionaries and grammar books, "
     "and the emergence of World Englishes and internet-influenced language change."),

    ("the ethics of genetic engineering and CRISPR technology",
     "Examine the moral and societal implications of CRISPR-Cas9 gene editing technology. Discuss "
     "therapeutic applications for diseases like sickle cell anemia, cystic fibrosis, and certain cancers. "
     "Address controversial uses including germline editing, designer babies, gene drives for eliminating "
     "malaria-carrying mosquitoes, and agricultural modifications. Explore ethical frameworks from "
     "utilitarianism to deontology applied to genetic engineering, regulatory approaches, and the "
     "distinction between treatment and enhancement."),

    ("the geopolitics of oil and energy resources in the modern world",
     "Analyze how control over petroleum reserves has shaped international relations since the early 20th "
     "century. Discuss the founding of OPEC, the 1973 oil crisis, the Gulf Wars, petrodollar system, "
     "fracking revolution in the United States, and Russia's use of energy as geopolitical leverage. "
     "Examine the transition toward renewable energy and how it may reshape global power dynamics, "
     "particularly for oil-dependent economies in the Middle East and Africa."),

    ("the biology of viruses and the history of pandemics",
     "Describe the structure and replication mechanisms of viruses, including RNA and DNA viruses, "
     "retroviruses, and bacteriophages. Survey major pandemics throughout history from the Black Death "
     "and the 1918 Spanish flu to HIV/AIDS, SARS, and COVID-19. Discuss how viruses jump between species "
     "through zoonotic spillover, the role of urbanization and global travel in pandemic spread, "
     "and lessons learned about public health preparedness and response."),

    ("the art and science of urban planning and smart cities",
     "Explore how cities are designed and how urban planning has evolved from ancient grid cities like "
     "Mohenjo-daro to modern smart city initiatives. Discuss zoning laws, transportation networks, "
     "green spaces, affordable housing challenges, and the concept of the 15-minute city. "
     "Examine how technology including IoT sensors, data analytics, and autonomous vehicles are "
     "being integrated into urban infrastructure, and address equity concerns in smart city development."),

    ("the mythology and religions of ancient Greece",
     "Explore the rich tapestry of Greek mythology including the Olympian gods led by Zeus, the Titans, "
     "heroic myths of Hercules, Perseus, and Odysseus, and tragic tales like those of Oedipus and Medea. "
     "Discuss the religious practices of ancient Greeks including temple worship, oracles at Delphi, "
     "mystery cults at Eleusis, and the Panhellenic games. Examine how Greek myths influenced Roman "
     "religion, Western literature, psychology through Freud and Jung, and modern popular culture."),

    ("the technology behind modern electric vehicles and battery science",
     "Examine the engineering of electric vehicles from lithium-ion battery chemistry to electric motor "
     "design, regenerative braking, and thermal management systems. Discuss the evolution from early EVs "
     "to Tesla's disruption of the auto industry, charging infrastructure challenges, solid-state battery "
     "research, and the environmental lifecycle analysis comparing EVs to internal combustion vehicles. "
     "Address supply chain concerns for lithium, cobalt, and rare earth minerals."),

    ("the history and impact of the printing press on civilization",
     "Analyze Johannes Gutenberg's invention of movable type printing around 1440 and its revolutionary "
     "effects on European society. Discuss how it enabled the Protestant Reformation by spreading Martin "
     "Luther's ideas, democratized knowledge beyond the clergy and nobility, accelerated scientific "
     "progress during the Enlightenment, and laid groundwork for modern journalism and mass media. "
     "Compare the printing revolution to the modern digital information revolution."),

    ("the oceanography of the Arctic and its role in global climate",
     "Investigate the unique oceanographic features of the Arctic Ocean including sea ice dynamics, "
     "thermohaline circulation, and the relationship between Arctic warming and global weather patterns. "
     "Discuss the concept of Arctic amplification, the opening of the Northwest Passage, impacts on "
     "indigenous communities, geopolitical competition for Arctic resources, and how rapidly declining "
     "sea ice serves as an early warning indicator for global climate change."),

    ("the philosophy and practice of mindfulness meditation",
     "Trace the origins of mindfulness from Buddhist Vipassana traditions to its secular adaptation in "
     "Western psychology through Jon Kabat-Zinn's MBSR program. Discuss neuroscientific research showing "
     "changes in brain structure and function from regular meditation practice, including effects on the "
     "prefrontal cortex, amygdala, and default mode network. Examine clinical evidence for treating "
     "anxiety, depression, chronic pain, and PTSD, along with critiques of the mindfulness industry."),

    ("the engineering challenges of building skyscrapers and supertall structures",
     "Explore how engineers design buildings that reach over 300 meters into the sky, from foundation "
     "engineering in various soil conditions to wind load analysis, seismic resistance, and the tuned "
     "mass damper systems used in structures like Taipei 101. Discuss the evolution from the Home "
     "Insurance Building in Chicago to the Burj Khalifa, materials science innovations in high-strength "
     "concrete and steel, elevator technology, and the race to build the next tallest building."),

    ("the anthropology of ancient Mesopotamian civilizations",
     "Examine the cradle of civilization between the Tigris and Euphrates rivers, covering Sumer, Akkad, "
     "Babylon, and Assyria. Discuss the invention of cuneiform writing, the Code of Hammurabi as one of "
     "the earliest legal codes, the ziggurat temples, irrigation agriculture, the Epic of Gilgamesh as "
     "the oldest surviving literary work, advances in mathematics including base-60 numbering, "
     "and the lasting influence on subsequent civilizations and the modern world."),

    ("the science of nutrition and the debate over optimal human diets",
     "Survey the history of nutritional science from the discovery of vitamins to modern metabolomics. "
     "Discuss macronutrients and micronutrients, the role of the gut microbiome, and conflicting dietary "
     "recommendations including low-fat, low-carb, ketogenic, Mediterranean, and plant-based diets. "
     "Examine the evidence for and against processed food consumption, intermittent fasting, nutritional "
     "epidemiology limitations, and how cultural and economic factors shape dietary choices globally."),

    ("the political history of democracy from Athens to the modern era",
     "Trace the evolution of democratic governance from Athenian direct democracy through the Roman "
     "Republic, medieval parliamentary developments in England, Enlightenment political philosophy of "
     "Locke, Montesquieu, and Rousseau, the American and French Revolutions, the expansion of suffrage "
     "to include women and minorities, decolonization movements, and the challenges facing democracy "
     "in the 21st century including populism, disinformation, and democratic backsliding."),

    ("the chemistry of cooking and molecular gastronomy",
     "Explain the chemical reactions that occur during cooking, including the Maillard reaction for "
     "browning, caramelization, protein denaturation, emulsification, and fermentation. Discuss how "
     "molecular gastronomy pioneers like Ferran Adria and Heston Blumenthal applied scientific techniques "
     "such as spherification, sous vide, liquid nitrogen freezing, and gelification to create innovative "
     "dishes. Examine how understanding food chemistry can improve everyday cooking and food safety."),

    ("the ecological impact of invasive species on native ecosystems",
     "Analyze how non-native species introduced through human activity have disrupted ecosystems worldwide. "
     "Discuss case studies including the brown tree snake in Guam, zebra mussels in the Great Lakes, "
     "cane toads in Australia, kudzu in the American South, and Asian carp threatening waterways. "
     "Examine the mechanisms of invasion, why some species become invasive while others don't, economic "
     "costs, biological control methods, and prevention strategies for managing invasive species."),

    ("the development of the internet from ARPANET to Web3",
     "Chronicle the evolution of the internet from the ARPANET military network in the 1960s through "
     "the invention of TCP/IP, the World Wide Web by Tim Berners-Lee, the browser wars, the dot-com "
     "boom and bust, Web 2.0 and social media, the mobile internet revolution, and current developments "
     "in Web3 including blockchain-based decentralized applications. Discuss net neutrality, digital "
     "divides, internet governance, and how the internet has transformed commerce, communication, and culture."),

    ("the astronomy of exoplanets and the search for extraterrestrial life",
     "Describe the methods astronomers use to detect exoplanets including the transit method, radial "
     "velocity, and direct imaging. Discuss discoveries by the Kepler and TESS missions, the concept of "
     "habitable zones, the diversity of planetary systems found, and the study of exoplanet atmospheres "
     "using the James Webb Space Telescope. Examine the Drake equation, the Fermi paradox, biosignatures, "
     "technosignatures, and SETI efforts in the search for intelligent life beyond Earth."),

    ("the history of feminism and the global fight for gender equality",
     "Survey the waves of feminism from the suffragette movements of the 19th century through second-wave "
     "feminism's focus on workplace equality and reproductive rights, third-wave intersectionality, and "
     "contemporary fourth-wave digital feminism including the MeToo movement. Discuss key thinkers like "
     "Mary Wollstonecraft, Sojourner Truth, Betty Friedan, bell hooks, and Chimamanda Ngozi Adichie. "
     "Examine ongoing challenges including the gender pay gap, political representation, and gender-based violence."),
]


def build_prompt(topic: str, details: str) -> str:
    """Build a unique, lengthy prompt to fill ~500 input tokens."""
    return (
        f"Write a detailed, comprehensive, and well-structured essay about {topic}. "
        f"Your essay should be thorough and cover multiple dimensions of the subject. "
        f"Specifically, please address the following aspects in depth:\n\n"
        f"{details}\n\n"
        f"Please organize your response with clear headings and subheadings, provide specific "
        f"examples and evidence where possible, and aim for a nuanced, balanced perspective "
        f"that considers multiple viewpoints. Your essay should be informative enough to serve "
        f"as a comprehensive introduction to this topic for an educated general audience."
    )


PROMPTS = [build_prompt(topic, details) for topic, details in ESSAY_TOPICS]


def send_request(request_id: int, prompt: str, results: list):
    """Send one streaming request, measure TTFT and total time."""
    url = f"{HOST}/v1/completions"
    payload = {
        "model": MODEL,
        "prompt": prompt,
        "max_tokens": MAX_TOKENS,
        "stream": True,
        "temperature": 0.7,
    }

    token_times = []
    tokens_received = 0
    error = None
    request_start = time.perf_counter()

    try:
        resp = requests.post(
            url, json=payload,
            headers={"Content-Type": "application/json"},
            stream=True,
            timeout=(60, 600),
        )
        resp.raise_for_status()

        for line in resp.iter_lines():
            if not line:
                continue
            line_str = line.decode()
            if not line_str.startswith("data: "):
                continue
            data_str = line_str[6:]
            if data_str.strip() == "[DONE]":
                break
            try:
                chunk = json.loads(data_str)
                text = chunk.get("choices", [{}])[0].get("text", "")
                if text:
                    token_times.append(time.perf_counter())
                    tokens_received += 1
            except json.JSONDecodeError:
                continue

    except Exception as e:
        error = str(e)

    request_end = time.perf_counter()

    if error or not token_times:
        results[request_id] = {
            "request_id": request_id,
            "error": error or "no tokens received",
            "total_sec": round(request_end - request_start, 3),
            "tokens": 0,
        }
        return

    ttft = token_times[0] - request_start
    total = request_end - request_start
    tok_per_sec = tokens_received / total if total > 0 else 0

    results[request_id] = {
        "request_id": request_id,
        "ttft_sec": round(ttft, 4),
        "total_sec": round(total, 3),
        "tokens": tokens_received,
        "tok_per_sec": round(tok_per_sec, 2),
        "error": None,
    }


def fetch_server_logs():
    """SSH into the server and grab recent vLLM logs looking for preemption/queuing signals."""
    if not SSH_KEY or not SSH_HOST:
        return {"raw": "(SSH not configured)", "waiting_max": 0, "preempted": 0, "kv_peak_pct": 0}

    signals = {"raw": "", "waiting_max": 0, "preempted": 0, "kv_peak_pct": 0}

    try:
        # Get last 200 lines of vLLM logs (journalctl or docker)
        cmd = [
            "ssh", "-i", SSH_KEY, "-o", "StrictHostKeyChecking=no",
            "-o", "ConnectTimeout=10",
            SSH_HOST,
            # Try journalctl first, then docker logs
            "sudo journalctl -u vllm --no-pager -n 200 2>/dev/null || "
            "sudo docker logs $(sudo docker ps -q | head -1) --tail 200 2>/dev/null || "
            "tail -200 /var/log/vllm*.log 2>/dev/null || "
            "echo 'NO_LOGS_FOUND'"
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        log_text = result.stdout + result.stderr
        signals["raw"] = log_text[-3000:] if len(log_text) > 3000 else log_text  # last 3k chars

        # Parse for specific signals
        for line in log_text.split("\n"):
            line_lower = line.lower()
            # Look for "Waiting: X" where X > 0
            if "waiting:" in line_lower:
                try:
                    parts = line.split("Waiting:")
                    if len(parts) > 1:
                        val = int(parts[1].strip().split()[0].strip(",").strip("."))
                        signals["waiting_max"] = max(signals["waiting_max"], val)
                except (ValueError, IndexError):
                    pass

            # Look for preemption
            if "preempt" in line_lower or "evict" in line_lower:
                signals["preempted"] += 1

            # Look for KV cache usage
            if "kv cache" in line_lower and "%" in line:
                try:
                    # Extract percentage
                    pct_idx = line.index("%")
                    num_str = ""
                    i = pct_idx - 1
                    while i >= 0 and (line[i].isdigit() or line[i] == "."):
                        num_str = line[i] + num_str
                        i -= 1
                    if num_str:
                        pct = float(num_str)
                        signals["kv_peak_pct"] = max(signals["kv_peak_pct"], pct)
                except (ValueError, IndexError):
                    pass

    except subprocess.TimeoutExpired:
        signals["raw"] = "(SSH timeout)"
    except Exception as e:
        signals["raw"] = f"(SSH error: {e})"

    return signals


def percentile(data, pct):
    """Simple percentile calculation."""
    if not data:
        return 0
    sorted_data = sorted(data)
    idx = int(len(sorted_data) * pct / 100)
    idx = min(idx, len(sorted_data) - 1)
    return sorted_data[idx]


def run_round(n: int):
    """Run one concurrency round with N simultaneous requests."""
    print(f"\n{'='*80}")
    print(f"  ROUND: N={n} concurrent requests (max_tokens={MAX_TOKENS})")
    print(f"{'='*80}")

    results = [None] * n
    threads = []

    # Pick N unique prompts (cycle if n > 40)
    round_prompts = [PROMPTS[i % len(PROMPTS)] for i in range(n)]

    round_start = time.perf_counter()

    for i in range(n):
        t = threading.Thread(target=send_request, args=(i, round_prompts[i], results))
        threads.append(t)

    # Launch all threads at once
    for t in threads:
        t.start()

    # Wait for all to finish
    for t in threads:
        t.join()

    round_end = time.perf_counter()
    round_duration = round_end - round_start

    # Collect metrics
    ok_results = [r for r in results if r and r.get("error") is None]
    fail_results = [r for r in results if r and r.get("error") is not None]
    no_result = [i for i, r in enumerate(results) if r is None]

    ttfts = [r["ttft_sec"] for r in ok_results]
    totals = [r["total_sec"] for r in ok_results]
    token_counts = [r["tokens"] for r in ok_results]
    tok_rates = [r["tok_per_sec"] for r in ok_results]

    # Print per-request details
    print(f"\n  Per-request details:")
    for r in sorted(results, key=lambda x: x["request_id"] if x else 999):
        if r is None:
            continue
        if r.get("error"):
            print(f"    req={r['request_id']:3d} | FAIL: {r['error'][:80]}")
        else:
            print(f"    req={r['request_id']:3d} | TTFT={r['ttft_sec']:.3f}s | "
                  f"Total={r['total_sec']:.1f}s | Tokens={r['tokens']} | "
                  f"{r['tok_per_sec']:.1f} tok/s")

    # Fetch server logs
    print(f"\n  Fetching server logs...")
    log_signals = fetch_server_logs()

    # Summary line
    ttft_p50 = percentile(ttfts, 50) if ttfts else 0
    ttft_max = max(ttfts) if ttfts else 0
    total_p50 = percentile(totals, 50) if totals else 0
    total_max = max(totals) if totals else 0

    summary = (
        f"N={n:2d} | OK: {len(ok_results):2d} Fail: {len(fail_results):2d} | "
        f"TTFT p50: {ttft_p50:.3f}s max: {ttft_max:.3f}s | "
        f"Total p50: {total_p50:.0f}s max: {total_max:.0f}s | "
        f"KV peak: {log_signals['kv_peak_pct']:.0f}% | "
        f"Waiting: {log_signals['waiting_max']} | "
        f"Preempted: {log_signals['preempted']}"
    )
    print(f"\n  >>> {summary}")

    # Print relevant log lines
    if log_signals["raw"] and log_signals["raw"] not in ("(SSH not configured)", "(SSH timeout)"):
        interesting_lines = []
        for line in log_signals["raw"].split("\n"):
            ll = line.lower()
            if any(kw in ll for kw in ["waiting:", "preempt", "evict", "kv cache", "error", "oom", "running:"]):
                interesting_lines.append(line.strip())
        if interesting_lines:
            print(f"\n  Interesting server log lines:")
            for il in interesting_lines[-20:]:
                print(f"    >> {il}")

    return {
        "n": n,
        "ok": len(ok_results),
        "fail": len(fail_results),
        "round_duration_sec": round(round_duration, 1),
        "ttft_p50": round(ttft_p50, 4),
        "ttft_max": round(ttft_max, 4),
        "ttft_avg": round(statistics.mean(ttfts), 4) if ttfts else 0,
        "total_p50": round(total_p50, 1),
        "total_max": round(total_max, 1),
        "total_avg": round(statistics.mean(totals), 1) if totals else 0,
        "avg_tokens": round(statistics.mean(token_counts), 0) if token_counts else 0,
        "avg_tok_per_sec": round(statistics.mean(tok_rates), 2) if tok_rates else 0,
        "kv_peak_pct": log_signals["kv_peak_pct"],
        "waiting_max": log_signals["waiting_max"],
        "preempted_count": log_signals["preempted"],
        "summary": summary,
        "errors": [r["error"] for r in fail_results],
        "log_excerpt": log_signals["raw"][-1000:] if log_signals["raw"] else "",
    }


def main():
    global HOST, SSH_KEY, SSH_HOST

    parser = argparse.ArgumentParser(description="Stage 2 Experiment 1: Preemption & Queuing Test")
    parser.add_argument("--host", required=True, help="vLLM server URL, e.g. http://44.201.96.75:8000")
    parser.add_argument("--ssh-key", default="~/.ssh/vllm-lab-key-2.pem", help="SSH private key path")
    parser.add_argument("--ssh-host", default="ubuntu@44.201.96.75", help="SSH user@host for log access")
    parser.add_argument("--levels", default=None, help="Comma-separated concurrency levels (default: 10,20,30,35,38,40,45,50)")
    args = parser.parse_args()

    HOST = args.host.rstrip("/")
    SSH_KEY = args.ssh_key.replace("~", str(__import__("pathlib").Path.home()))
    SSH_HOST = args.ssh_host

    if args.levels:
        levels = [int(x.strip()) for x in args.levels.split(",")]
    else:
        levels = CONCURRENCY_LEVELS

    print(f"vLLM Preemption Experiment")
    print(f"Host: {HOST}")
    print(f"Model: {MODEL}")
    print(f"Max tokens: {MAX_TOKENS}")
    print(f"Unique prompts: {len(PROMPTS)}")
    print(f"Concurrency levels: {levels}")
    print(f"SSH: {SSH_HOST} (key: {SSH_KEY})")

    # Quick health check
    try:
        r = requests.get(f"{HOST}/v1/models", timeout=10)
        models = r.json()
        print(f"Server models: {json.dumps(models.get('data', [{}])[0].get('id', 'unknown'))}")
    except Exception as e:
        print(f"WARNING: Health check failed: {e}")

    all_results = []

    for n in levels:
        result = run_round(n)
        all_results.append(result)

        # Save after each round
        out_path = "/Users/mrefaat/Apps/LLM_Hands_On/vLLM_Inference/stage2_exp1_preemption_results.json"
        with open(out_path, "w") as f:
            json.dump(all_results, f, indent=2)

        # Sleep between rounds to let vLLM drain
        if n != levels[-1]:
            print(f"\n  Sleeping 15 seconds to let vLLM drain...")
            time.sleep(15)

    # Final summary table
    print(f"\n\n{'='*100}")
    print(f"  FINAL SUMMARY")
    print(f"{'='*100}")
    for r in all_results:
        print(f"  {r['summary']}")

    print(f"\nResults saved to: {out_path}")


if __name__ == "__main__":
    main()
