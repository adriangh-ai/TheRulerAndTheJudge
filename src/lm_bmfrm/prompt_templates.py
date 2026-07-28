from textwrap import dedent


class PromptTemplate:
    def __init__(self, template_str, dimension_defs):
        self.template_str = template_str
        self.dimension_defs = dimension_defs

    def format(self, **kwargs):
        raise NotImplementedError("Subclasses must implement the format method.")


class SummevallTemplate(PromptTemplate):
    DIMS = ["coherence", "consistency", "fluency", "relevance"]

    def __init__(self):
        template_str = dedent(
            """\
            Task: Evaluate the quality of summaries written for a news article.

            Instructions (steps):
            1) Carefully read the news article; be aware of the information it contains.
            2) Read the proposed summary.
            3) Rate the summary on a scale from 1 (worst) to 5 (best) for "{dimension}" according to its definition.

            Definition of "{dimension}":
            {dimension_defs}

            Scale:
            - 1 = worst, 5 = best (no additional textual anchors for intermediate scores are provided).

            Return the numerical score in a json object with the following format AFTER your reasoning: {{"score": <score>}}

            Article:
            {article}
            Summary:
            {summary}
        """
        )
        dimension_defs = {
            "relevance": (
                "- Measures how well the summary captures the key points of the article.\n"
                "- Consider whether all and only the important aspects are contained in the summary."
            ),
            "consistency": (
                "- Measures whether the facts in the summary are consistent with the facts in the original article.\n"
                "- Consider whether the summary reproduces all facts accurately and does not make up untrue information."
            ),
            "fluency": (
                "- Measures the quality of individual sentences (whether they are well-written and grammatically correct).\n"
                "- Consider the quality of individual sentences."
            ),
            "coherence": (
                "- Measures the quality of all sentences collectively (whether they fit together and sound natural).\n"
                "- Consider the quality of the summary as a whole."
            ),
        }
        super().__init__(template_str, dimension_defs)

    def format(self, dimension, **kwargs):
        article = kwargs.get("question")
        summary = kwargs.get("response")
        if article is None or summary is None:
            raise ValueError(
                "Both 'article' and 'summary' must be provided to format the SummevallTemplate."
            )
        
        context = self.template_str.format(
            dimension=dimension, 
            dimension_defs=self.dimension_defs[dimension],
            article = article,
            summary = summary,
        )
        messages = [
            {"role": "system", "content": "system_prompt"},
            {"role": "user", "content": context},
        ]
        return messages
    

class SummevallV2Template(PromptTemplate):
    DIMS = ["coherence", "consistency", "fluency", "relevance"]

    def __init__(self):
        template_str = dedent(
            """\
            You will be given one summary written for a news article.

            Your task is to rate the summary on one metric.

            Please make sure you read and understand these instructions carefully. Please keep this document open while reviewing, and refer to it as needed.

            Evaluation Criteria: 
            {dimension} (1-5) - {dimension_defs}
            
            Evaluation Steps:
            {dimension_steps}

            Source Text:
            {article}
            Summary:
            {summary}

            Evaluation Form (scores ONLY):
            - {dimension}:"""
        )
        dimension_defs = {
            "relevance": (
                "selection of important content from the source. The summary should include only important information from the source document. Annotators were instructed to penalize summaries which contained redundancies and excess information."
            ),
            "consistency": (
                "the factual alignment between the summary and the summarized source. A factually consistent summary contains only statements that are entailed by the source document. Annotators were also asked to penalize summaries that contained hallucinated facts."
            ),
            "fluency": (
                "the quality of the summary in terms of grammar, spelling, punctuation, word choice, and sentence structure."
            ),
            "coherence": (
                "the collective quality of all sentences. We align this dimension with the DUC quality question of structure and coherence whereby \"the summary should be well-structured and well-organized. The summary should not just be a heap of related information, but should build from sentence to a coherent body of information about a topic.\""
            ),
        }
        self.dimension_steps = {
            "relevance": (
                "1. Read the summary and the source document carefully.\n"
                "2. Compare the summary to the source document and identify the main points of the article.\n"
                "3. Assess how well the summary covers the main points of the article, and how much irrelevant or redundant information it contains.\n"
                "4. Assign a score for relevance on a scale of 1 to 5, where 1 is the lowest and 5 is the highest based on the Evaluation Criteria."
            ),
            "consistency": (
                "1. Read the news article carefully and identify the main facts and details it presents.\n"
                "2. Read the summary and compare it to the article. Check if the summary contains any factual errors that are not supported by the article.\n"
                "3. Assign a score for consistency on a scale of 1 to 5, where 1 is the lowest and 5 is the highest based on the Evaluation Criteria."
            ),
            "fluency": (
                "1. Read the summary carefully and evaluate its grammar, spelling, punctuation, word choice, and sentence structure.\n"
                "2. Assess how well the sentences flow together and whether the summary is easy to read and understand.\n"
                "3. Assign a score for fluency on a scale of 1 to 5, where 1 is the lowest and 5 is the highest based on the Evaluation Criteria."
            ),
            "coherence": (
                "1. Read the news article carefully and identify the main topic and key points.\n"
                "2. Read the summary and compare it to the news article. Check if the summary covers the main topic and key points of the news article, and if it presents them in a clear and logical order.\n"
                "3. Assign a score for coherence on a scale of 1 to 5, where 1 is the lowest and 5 is the highest based on the Evaluation Criteria."
            ),
        }
        super().__init__(template_str, dimension_defs)

    def format(self, dimension, **kwargs):
        article = kwargs.get("question")
        summary = kwargs.get("response")
        if article is None or summary is None:
            raise ValueError(
                "Both 'article' and 'summary' must be provided to format the SummevallV2Template."
            )

        context = self.template_str.format(
            dimension=dimension,
            dimension_defs=self.dimension_defs[dimension],
            dimension_steps=self.dimension_steps[dimension],
            article=article,
            summary=summary,
        )
        messages = [
            {"role": "system", "content": "system_prompt"},
            {"role": "user", "content": context},
        ]
        return messages


class HannaTemplate(PromptTemplate):
    """
    HANNA-style LLM-as-a-judge prompt template.

    This version is aligned with the 2024 TACL HANNA LLM evaluation setup:
    - supports Eval-Prompt 1..4
    - uses a plain prompt surface form closer to Figure 3
    - does NOT request JSON output
    - returns a single user message
    - uses the original HANNA criteria definitions as guidelines for Eval-Prompt 3

    Eval-Prompt variants:
      1 -> simple rating
      2 -> rating + explanation
      3 -> rating + explanation + guidelines
      4 -> rating + explanation + human story reference
    """

    DIMS = [
        "Relevance",
        "Coherence",
        "Empathy",
        "Surprise",
        "Engagement",
        "Complexity",
    ]

    def __init__(self, build_eval_prompt=3):
        template_str = build_eval_prompt
        dimension_defs = {
            "Relevance": (
                "Measures how well the story matches its prompt.\n"
                "- 1: The story has no relationship with the prompt at all.\n"
                "- 2: The story only has a weak relationship with the prompt.\n"
                "- 3: The story roughly matches the prompt.\n"
                "- 4: The story matches the prompt, except for one or two small aspects.\n"
                "- 5: The story matches the prompt exactly."
            ),
            "Coherence": (
                "Measures whether the story makes sense (logic, plot, setting consistency).\n"
                "- 1: The story does not make sense at all. For instance, the setting and/or characters keep changing, "
                "and/or there is no understandable plot.\n"
                "- 2: Most of the story does not make sense.\n"
                "- 3: The story mostly makes sense but has some incoherences.\n"
                "- 4: The story almost makes sense overall, except for one or two small incoherences.\n"
                "- 5: The story makes sense from beginning to end."
            ),
            "Empathy": (
                "Measures how well you understood the characters' emotions "
                "(regardless of whether you agreed with them).\n"
                "- 1: The characters seemed apathetic to you.\n"
                "- 2: At least one character slightly related to you on an emotional level.\n"
                "- 3: You recognized specific, but not necessarily strong, emotions "
                "(e.g. sadness, joy, fear...) in at least one character.\n"
                "- 4: At least one character emotionally involved you, but minor details "
                "prevented you from completely relating to them.\n"
                "- 5: At least one character completely involved you on an emotional level."
            ),
            "Surprise": (
                "Measures how surprising the end of the story was.\n"
                "- 1: The ending seemed completely obvious from the start, or doesn't make any sense at all.\n"
                "- 2: The ending was easily predictable after a few sentences.\n"
                "- 3: The ending was predictable after half of the story.\n"
                "- 4: The ending surprised you, but would have been difficult to predict.\n"
                "- 5: The ending surprised you, and still seemed as if it could very reasonably have been predicted "
                "(i.e., there were enough clues in the story)."
            ),
            "Engagement": (
                "Measures how much you engaged with the story.\n"
                "- 1: You found the story boring and were glad it was over.\n"
                "- 2: You found one or two things interesting in the story, but no more.\n"
                "- 3: The story was mildly interesting.\n"
                "- 4: The story almost kept you engaged until the end.\n"
                "- 5: You were so engaged that you wished there was a sequel."
            ),
            "Complexity": (
                "Measures how elaborate the story is (vocabulary, world-building, plot intricacy).\n"
                "- 1: The setting of the story is extremely simple; it only involves one or two characters or concepts.\n"
                "- 2: The setting of the story is simple; one or two characters, a simple plot, maybe an indication "
                "of time or location.\n"
                "- 3: The story is somewhat developed: it involves at least one of the following: complex concepts, "
                "realistic characters, an intricate plot, an underlying history or circumstances, precise descriptions.\n"
                "- 4: The story is developed: it involves at least two of the following components (listed in score 3).\n"
                "- 5: The story is well thought-out: it involves at least three of the following components "
                "(listed in score 3)."
            ),
        }

        super().__init__(template_str=template_str, dimension_defs=dimension_defs)

    def _validate_dimension(self, dimension: str) -> str:
        if dimension not in self.DIMS:
            raise ValueError(
                f"Unknown dimension '{dimension}'. Valid dimensions: {self.DIMS}"
            )
        return dimension

    def _build_eval_prompt_1(self, dimension: str, prompt: str, story: str) -> str:
        return dedent(
            f"""\
            Prompt: {prompt}

            Target Story: {story}

            Rate the story on a scale from 1 to 5 on {dimension}. Rating:"""
        )

    def _build_eval_prompt_2(self, dimension: str, prompt: str, story: str) -> str:
        return dedent(
            f"""\
            Prompt: {prompt}

            Target Story: {story}

            Rate the story on a scale from 1 to 5 on {dimension} and explain your answer. Rating:"""
        )

    def _build_eval_prompt_3(self, dimension: str, prompt: str, story: str) -> str:
        guidelines = self.dimension_defs[dimension]
        return dedent(
            f"""\
            Prompt: {prompt}

            Target Story: {story}

            Guidelines:
            {guidelines}

            Rate the story on a scale from 1 to 5 on {dimension} and explain your answer. Use the provided guidelines. Rating:"""
        )

    def _build_eval_prompt_4(
        self,
        dimension: str,
        prompt: str,
        story: str,
        human_story: str,
    ) -> str:
        return dedent(
            f"""\
            Prompt: {prompt}

            Target Story: {story}

            Human Story: {human_story}

            Rate the target story on a scale from 1 to 5 on {dimension} and explain your answer. Do not rate the human story; it is here only for reference. Rating:"""
        )

    def format(self, dimension, **kwargs):
        """
        Required kwargs:
            prompt: the writing prompt / story-prompt
            story: the target story to evaluate

        Optional kwargs:
            eval_prompt_version: one of {1, 2, 3, 4}; default=3
            human_story: required only for eval_prompt_version=4

        Returns:
            list[dict]: chat-style messages
        """
        dimension = self._validate_dimension(dimension)

        prompt = (kwargs.get("Prompt") or kwargs.get("question"))
        story = (kwargs.get("Story") or kwargs.get("response"))
        eval_prompt_version = self.template_str
        human_story = kwargs.get("human_story")

        if prompt is None or story is None:
            raise ValueError(
                "Both 'prompt' and 'story' must be provided to format the HannaTemplate."
            )

        if eval_prompt_version == 1:
            context = self._build_eval_prompt_1(dimension, prompt, story)
        elif eval_prompt_version == 2:
            context = self._build_eval_prompt_2(dimension, prompt, story)
        elif eval_prompt_version == 3:
            context = self._build_eval_prompt_3(dimension, prompt, story)
        elif eval_prompt_version == 4:
            if human_story is None:
                raise ValueError(
                    "'human_story' must be provided when eval_prompt_version=4."
                )
            context = self._build_eval_prompt_4(
                dimension, prompt, story, human_story
            )
        else:
            raise ValueError(
                "eval_prompt_version must be one of {1, 2, 3, 4}."
            )

        messages = [
            {"role": "user", "content": context},
        ]
        return messages
    

class HitzBasseTemplate(PromptTemplate):
    """
    Faithful reproduction of the HITZ-BASS LLM-as-a-judge annotation setup
    using Prometheus-Eval's ABSOLUTE_PROMPT_WO_REF + SCORE_RUBRIC_TEMPLATE.

    Important:
    - This reproduces annotation, not summary generation.
    - It evaluates one dimension at a time.
    - It does NOT request JSON output, because that would no longer be an
      exact reproduction of their prompting protocol.
    """

    DIMS = ["coherence", "consistency", "fluency", "relevance", "5w1h"]

    SCORE_RUBRIC_TEMPLATE = dedent(
        """\
        [{criteria}]
        Score 1: {score1_description}
        Score 2: {score2_description}
        Score 3: {score3_description}
        Score 4: {score4_description}
        Score 5: {score5_description}""".strip()
    )

    ABSOLUTE_PROMPT_WO_REF = dedent(
        """\
        ###Task Description:
        An instruction (might include an Input inside it), a response to evaluate, and a score rubric representing a evaluation criteria are given.
        1. Write a detailed feedback that assess the quality of the response strictly based on the given score rubric, not evaluating in general.
        2. After writing a feedback, write a score that is an integer between 1 and 5. You should refer to the score rubric.
        3. The output format should look as follows: "(write a feedback for criteria) [RESULT] (an integer number between 1 and 5)"
        4. Please do not generate any other opening, closing, and explanations.

        ###The instruction to evaluate:
        {instruction}

        ###Response to evaluate:
        {response}

        ###Score Rubrics:
        {rubric}

        ###Feedback: """
    )

    RUBRICS = {
        "coherence": {
            "criteria": (
                "Coherence - the collective quality of all sentences. "
                "We align this dimension with the DUC quality question (Dang, 2005) "
                "of structure and coherence whereby 'the summary should be "
                "well-structured and well-organized. The summary should not just be "
                "a heap of related information, but should build from sentence to "
                "sentence to a coherent body of information about a topic.' "
                "To clarify, if you were a language teacher, how would you grade "
                "the flow of ideas? Does the summary contain connectors to conect ideas? "
                "It doesn't matter if there are grammatical errors or incorrect information. "
                "If the summary is simply a list of events, we penalize it. "
                "We do not penalize single sentences for not having connectors."
            ),
            "score1_description": (
                "The summary contains a bulletpoint list of events and there is no "
                "internal consistency (e.g., random words)"
            ),
            "score2_description": (
                "The summary contains a bulletpoint list of events, but there is no "
                "consistency between the phrases in each bulletpoint "
                "(e.g., 1. Keyboard, 2. Watching TV)"
            ),
            "score3_description": (
                "The summary contains a bulletpoint list of events, pero the phrases "
                "in each bullet point are developed well."
            ),
            "score4_description": (
                "The summary contains implicit coherence, but not explicit. "
                "That means that there are no temporal connectors or discourse markers, "
                "but the sentences have a temporal/causal flow."
            ),
            "score5_description": (
                "The summary contains explicit coherence via textual resources such as "
                "temporal connector or discourse markers that mark the relationship "
                "between ideas and sentences"
            ),
        },
        "consistency": {
            "criteria": (
                "Consistency - the factual alignment between the summary and the "
                "summarized source. A factually consistent summary contains only "
                "statements that are entailed by the source document. Annotators "
                "were also asked to penalize summaries that contained hallucinated facts. "
                "That is to say, does the summary hallucinate or make up information? "
                "If the summary contains information not found in the original document, "
                "we penalize it. For temporal expressions (today, yesterday, this year), "
                "if the expression is consistent with the original information, we assume "
                "that the summary is consistent and do not penalize."
            ),
            "score1_description": "The summary does not contain any ideas from the original text.",
            "score2_description": "The summary contains a large amount of incorrect information.",
            "score3_description": "The summary contains several incorrect pieces of information.",
            "score4_description": "The summary contains once incorrect piece of information.",
            "score5_description": "The summary is completely factual.",
        },
        "fluency": {
            "criteria": (
                "Fluency - the quality of individual sentences. Drawing again from "
                "the DUC quality guidelines, sentences in the summary 'should have "
                "no formatting problems, capitalization errors or obviously "
                "ungrammatical sentences (e.g., fragments, missing components) that "
                "make the text difficult to read.' Is the summary well written "
                "following the grammar and spelling conventions for the language? "
                "If it has errors, we penalize it. We only take grammar and fluency "
                "into account. While the style can sometimes be forced, if it is "
                "grammatically correct, we give full points."
            ),
            "score1_description": (
                "There are many important grammatical errors or the summary mixes "
                "language varieties/dialects in a very unnatural and uncomfortable way "
                "or the summary is not written in the language it is asked for."
            ),
            "score2_description": (
                "There are various important grammatical errors or the summary mixes "
                "language varieties/dialects in an unnatural and uncomfortable way."
            ),
            "score3_description": (
                "There are grammatical errors of lesser importance or the summary mixes "
                "language varieties/dialects in a somewhat unnatural and uncomfortable way."
            ),
            "score4_description": "There are few grammatical errors.",
            "score5_description": (
                "There is no grammatical or spelling error and the variety of the "
                "language/dialect used is coherent."
            ),
        },
        "relevance": {
            "criteria": (
                "Relevance - selection of important content from the source. "
                "The summary should include only important information from the source document. "
                "Penalize summaries that contain redundancies and excess information. "
                "We also penalize phrases metalinguistic phrases generated by the model, "
                "such as 'Here is the summary:' (-1). Finally, we also penalize the "
                "presence of subjective opinions if they are not part of the original text (-1)"
            ),
            "score1_description": (
                "The summary is the original text or only contains irrelevant information."
            ),
            "score2_description": (
                "The summary contains a large amount of irrelevant information."
            ),
            "score3_description": (
                "The summary contains a mix of relevant and irrelevant information."
            ),
            "score4_description": "The summary contains mostly relevant information.",
            "score5_description": "All information in the summary is relevant.",
        },
        "5w1h": {
            "criteria": (
                "5W1H - maintenance of all important information (the Ws in 5W1H - "
                "who, what, when, where, why, how) from the source document. "
                "The summary should not lack any important information available in the source"
            ),
            "score1_description": (
                "The summary contains no relevant W from the original text"
            ),
            "score2_description": "The summary contains only 1-2 Ws",
            "score3_description": "The summary lacks several relevant Ws",
            "score4_description": "The summary is lacking one relevant W.",
            "score5_description": (
                "The summary contains all the relevant Ws from the original text."
            ),
        },
    }

    DIMENSION_ALIASES = {
        "coherence": "coherence",
        "consistency": "consistency",
        "fluency": "fluency",
        "relevance": "relevance",
        "5w1h": "5w1h",
        "5W1H": "5w1h",
    }

    def __init__(self):
        super().__init__(self.ABSOLUTE_PROMPT_WO_REF, self.RUBRICS)

    def _normalize_dimension(self, dimension: str) -> str:
        if dimension not in self.DIMENSION_ALIASES:
            raise ValueError(
                f"Unknown dimension '{dimension}'. Valid dimensions: {list(self.DIMENSION_ALIASES.keys())}"
            )
        return self.DIMENSION_ALIASES[dimension]

    def _build_rubric(self, dimension: str) -> str:
        d = self.dimension_defs[dimension]
        return self.SCORE_RUBRIC_TEMPLATE.format(
            criteria=d["criteria"],
            score1_description=d["score1_description"],
            score2_description=d["score2_description"],
            score3_description=d["score3_description"],
            score4_description=d["score4_description"],
            score5_description=d["score5_description"],
        )

    def format(self, dimension, **kwargs):
        """
        Expected kwargs:
            question / article / source_document -> source text
            response / summary                 -> candidate summary

        To match the HITZ-BASS repo setup, the instruction is:
            "Summarize the following text:\n{article}"
        """
        dimension = self._normalize_dimension(dimension)

        article = (
            kwargs.get("question")
            or kwargs.get("article")
            or kwargs.get("document")
            or kwargs.get("item")
        )
        summary = kwargs.get("response") or kwargs.get("summary")

        if article is None or summary is None:
            raise ValueError(
                "Both source text and summary must be provided. "
                "Use question/article/source_document and response/summary."
            )

        instruction = f"Summarize the following text:\n{article}"
        rubric = self._build_rubric(dimension)

        context = self.template_str.format(
            instruction=instruction,
            response=summary,
            rubric=rubric,
        )

        # Single user message to preserve the final prompt text as closely as possible
        messages = [
            {"role": "user", "content": context},
        ]
        return messages


class HitzBasseSpanishTemplate(PromptTemplate):
    """
    Versión en español del setup de anotación LLM-as-a-judge de HITZ-BASS,
    manteniendo la estructura tipo Prometheus-Eval:
    - una dimensión por vez
    - feedback detallado + [RESULT] + puntuación
    - sin pedir JSON dentro del prompt
    """

    DIMS = ["coherence", "consistency", "fluency", "relevance", "5w1h"]

    SCORE_RUBRIC_TEMPLATE = dedent(
        """\
        [{criteria}]
        Puntuación 1: {score1_description}
        Puntuación 2: {score2_description}
        Puntuación 3: {score3_description}
        Puntuación 4: {score4_description}
        Puntuación 5: {score5_description}"""
    ).strip()

    ABSOLUTE_PROMPT_WO_REF = dedent(
        """\
        ###Descripción de la tarea:
        Se proporciona un artículo fuente y un resumen a evaluar y una rúbrica de puntuación que representa un criterio de evaluación.
        1. Escribe un feedback detallado que evalúe la calidad del resumen estrictamente en función de la rúbrica proporcionada, no de forma general.
        2. Después de escribir el feedback, escribe una puntuación entera entre 1 y 5. Debes basarte en la rúbrica.
        3. El formato de salida debe ser exactamente el siguiente: "(escribe aquí tu feedback) [RESULT] (un número entero entre 1 y 5)"
        4. No generes ninguna otra introducción, cierre ni explicación adicional.

        ###Artículo fuente:
        {instruction}

        ###Resumen a evaluar:
        {response}

        ###Rúbrica de puntuación:
        {rubric}

        ###Feedback: """
    )

    RUBRICS = {
        "coherence": {
            "criteria": (
                "Coherencia - calidad colectiva de todas las frases del resumen. "
                "Alineamos esta dimensión con la pregunta de calidad de DUC (Dang, 2005) "
                "sobre estructura y coherencia: el resumen debe estar bien estructurado y "
                "bien organizado; no debe ser un mero montón de información relacionada, "
                "sino que debe construir un cuerpo coherente de información de frase en frase. "
                "Dicho de forma más sencilla: si fueras profesor/a del idioma, ¿cómo puntuarías "
                "el flujo de ideas? ¿El resumen tiene conectores para relacionar ideas? "
                "No importa si hay fallos de información o de gramática. Si el resumen es "
                "simplemente una lista de eventos, se penaliza. No se penaliza una única frase "
                "por no tener conectores."
            ),
            "score1_description": (
                "Serie de eventos en formato de bullet points y sin coherencia interna "
                "(por ejemplo, palabras sueltas)."
            ),
            "score2_description": (
                "Solo hay bullet points y falta coherencia entre los sintagmas "
                "(por ejemplo, 1. Teclado, 2. Ver la tele)."
            ),
            "score3_description": (
                "Hay una lista de eventos o puntos, pero cada punto está desarrollado "
                "de forma aceptable."
            ),
            "score4_description": (
                "Existe coherencia implícita: no hay conectores temporales ni marcadores "
                "discursivos explícitos, pero las frases tienen flujo temporal o causal."
            ),
            "score5_description": (
                "Existe coherencia explícita: hay recursos textuales que marcan la relación "
                "entre ideas y frases, como conectores temporales o marcadores discursivos."
            ),
        },
        "consistency": {
            "criteria": (
                "Consistencia - alineación factual entre el resumen y el artículo fuente. "
                "Un resumen factualmente consistente contiene solo afirmaciones que se derivan "
                "del documento fuente. También se penalizan los resúmenes que contienen hechos "
                "alucinados. Es decir: ¿alucina o se inventa información? Si el resumen contiene "
                "información que no está en el artículo fuente, se penaliza. Para expresiones "
                "temporales relativas (hoy, ayer, este año), si la expresión es consistente con "
                "la información del artículo fuente, consideramos que el resumen es consistente y no "
                "penalizamos."
            ),
            "score1_description": "El resumen no contiene ninguna idea del artículo fuente.",
            "score2_description": "El resumen contiene muchos datos o información no veraz.",
            "score3_description": "El resumen contiene varios datos o piezas de información no veraz.",
            "score4_description": "El resumen contiene un único dato o pieza de información no veraz.",
            "score5_description": "El resumen no miente; es completamente factual.",
        },
        "fluency": {
            "criteria": (
                "Fluidez - calidad de las frases individuales. De acuerdo con las guías de "
                "calidad de DUC, las frases del resumen no deberían tener problemas de formato, "
                "errores de capitalización ni construcciones claramente agramaticales "
                "(por ejemplo, fragmentos o componentes ausentes) que dificulten la lectura. "
                "¿El resumen está escrito siguiendo la gramática y la ortografía del idioma? "
                "Si tiene fallos, se penaliza. Solo se tiene en cuenta la gramática, la ortografía "
                "y la adecuación lingüística. A veces el estilo puede sonar algo forzado, pero si "
                "es correcto, se considera adecuado."
            ),
            "score1_description": (
                "Hay muchos fallos gramaticales importantes, o se mezclan variedades o dialectos "
                "de manera muy poco natural e incómoda, o el resumen no está escrito en la lengua solicitada."
            ),
            "score2_description": (
                "Hay varios fallos gramaticales importantes, o se mezclan variedades o dialectos "
                "de manera poco natural e incómoda."
            ),
            "score3_description": (
                "Hay fallos gramaticales de menor importancia, o se mezclan variedades o dialectos "
                "de manera algo poco natural e incómoda."
            ),
            "score4_description": "Hay unos pocos fallos gramaticales.",
            "score5_description": (
                "No hay ningún fallo de gramática ni ortografía y se usa una única variedad "
                "coherente del idioma o dialecto."
            ),
        },
        "relevance": {
            "criteria": (
                "Relevancia - selección del contenido importante del artículo fuente. El resumen "
                "debe incluir solo información importante del artículo fuente. Se penalizan "
                "las redundancias y el exceso de información. Es decir: ¿el resumen contiene "
                "solo información relevante del artículo fuente? También se penalizan frases "
                "metalingüísticas generadas por el modelo, como '¿Te gustaría que elabore algún "
                "aspecto específico de este resumen?', porque no añaden información relevante "
                "(-1 en la nota final). Asimismo, se penaliza la presencia de valoraciones "
                "subjetivas si no forman parte del artículo fuente (-1 en la nota final). "
                "Si son noticias con varios hilos discursivos o listas de acontecimientos, "
                "debe priorizarse el hilo principal y solo incluir elementos del segundo hilo "
                "si aportan al principal."
            ),
            "score1_description": (
                "El resumen es prácticamente el artículo fuente o solo contiene información irrelevante."
            ),
            "score2_description": (
                "El resumen contiene mucha información irrelevante, aunque puede haber algo relevante."
            ),
            "score3_description": (
                "El resumen contiene una mezcla de información relevante e irrelevante."
            ),
            "score4_description": (
                "La mayoría de la información es relevante; puede incluir pocos elementos secundarios "
                "que no aportan demasiado."
            ),
            "score5_description": (
                "Toda la información del resumen es relevante y recoge lo esencial del hilo principal; "
                "si hay un segundo hilo, solo incluye lo importante cuando aporta al principal."
            ),
        },
        "5w1h": {
            "criteria": (
                "5W - mantenimiento de toda la información importante del artículo fuente. El resumen "
                "no debe omitir ninguna información importante disponible en el artículo fuente. "
                "La pregunta clave es: ¿falta alguna W relevante del artículo fuente en el resumen? "
                "Si hay muchas Ws, no se penaliza que falten algunas menos relevantes si están las "
                "principales. Cuando hay varios hilos, deben aparecer todas las Ws esenciales del "
                "hilo principal y las importantes del segundo hilo solo si aportan al principal."
            ),
            "score1_description": "El resumen no contiene ninguna W relevante del artículo fuente.",
            "score2_description": "El resumen contiene solo 1 o 2 Ws relevantes.",
            "score3_description": "El resumen contiene algunas Ws relevantes, pero faltan varias importantes.",
            "score4_description": "El resumen contiene casi todas las Ws relevantes, pero falta una importante.",
            "score5_description": (
                "El resumen contiene todas las Ws relevantes del artículo fuente, al menos las esenciales "
                "del hilo principal y las importantes del segundo hilo si aportan al principal."
            ),
        },
    }

    DIMENSION_ALIASES = {
        "coherence": "coherence",
        "consistency": "consistency",
        "fluency": "fluency",
        "relevance": "relevance",
        "5w1h": "5w1h",
        "5W1H": "5w1h",
        "5w": "5w1h",
        "5W": "5w1h",
    }

    def __init__(self):
        super().__init__(self.ABSOLUTE_PROMPT_WO_REF, self.RUBRICS)

    def _normalize_dimension(self, dimension: str) -> str:
        if dimension not in self.DIMENSION_ALIASES:
            raise ValueError(
                f"Unknown dimension '{dimension}'. Valid dimensions: {list(self.DIMENSION_ALIASES.keys())}"
            )
        return self.DIMENSION_ALIASES[dimension]

    def _build_rubric(self, dimension: str) -> str:
        d = self.dimension_defs[dimension]
        return self.SCORE_RUBRIC_TEMPLATE.format(
            criteria=d["criteria"],
            score1_description=d["score1_description"],
            score2_description=d["score2_description"],
            score3_description=d["score3_description"],
            score4_description=d["score4_description"],
            score5_description=d["score5_description"],
        )

    def format(self, dimension, **kwargs):
        """
        Expected kwargs:
            question / article / document / item -> texto fuente
            response / summary                   -> resumen candidato

        La instrucción sigue el patrón del template original, pero en español:
            "Resume el siguiente texto:\n{article}"
        """
        dimension = self._normalize_dimension(dimension)

        article = (
            kwargs.get("question")
            or kwargs.get("article")
            or kwargs.get("document")
            or kwargs.get("item")
        )
        summary = kwargs.get("response") or kwargs.get("summary")

        if article is None or summary is None:
            raise ValueError(
                "Both source text and summary must be provided. "
                "Use question/article/document/item and response/summary."
            )

        instruction = f"Resume el siguiente texto:\n{article}"
        rubric = self._build_rubric(dimension)

        context = self.template_str.format(
            instruction=instruction,
            response=summary,
            rubric=rubric,
        )

        messages = [
            {"role": "user", "content": context},
        ]
        return messages
    

class HitzBasseSummevalTemplate(PromptTemplate):
    """
    SummEval-style prompt, but using HITZ-BASS dimensions/definitions.

    This is intentionally NOT a faithful reproduction of HITZ-BASS prompting.
    It is a prompt-format ablation:
      - same HITZ-BASS dimensions
      - same source/summary evaluation task
      - SummEval-style instruction surface form
      - JSON score output
    """

    DIMS = ["coherence", "consistency", "fluency", "relevance", "5w1h"]

    DIMENSION_ALIASES = {
        "coherence": "coherence",
        "consistency": "consistency",
        "fluency": "fluency",
        "relevance": "relevance",
        "5w1h": "5w1h",
        "5W1H": "5w1h",
        "5w": "5w1h",
        "5W": "5w1h",
    }

    def __init__(self):
        template_str = dedent(
            """\
            Task: Evaluate the quality of a summary written for a source document.

            Instructions (steps):
            1) Carefully read the source document; be aware of the information it contains.
            2) Read the proposed summary.
            3) Rate the summary on a scale from 1 (worst) to 5 (best) for "{dimension}" according to its definition.

            Definition of "{dimension}":
            {dimension_defs}

            Scale:
            - 1 = worst, 5 = best (no additional textual anchors for intermediate scores are provided).

            Return the numerical score in a json object with the following format AFTER your reasoning: {{"score": <score>}}

            Source document:
            {article}

            Summary:
            {summary}
            """
        )

        dimension_defs = {
            "relevance": (
                "- Measures selection of important content from the source.\n"
                "- Consider whether the summary includes only important information from the source document.\n"
                "- Penalize redundancies, excess information, metalinguistic filler generated by the model, and subjective opinions not present in the source."
            ),
            "consistency": (
                "- Measures the factual alignment between the summary and the source document.\n"
                "- Consider whether the summary contains only information supported by the original text.\n"
                "- Penalize hallucinated or invented information."
            ),
            "fluency": (
                "- Measures the quality of individual sentences.\n"
                "- Consider grammar, spelling, and whether the summary follows the conventions of the target language.\n"
                "- Focus on sentence-level linguistic quality rather than factual correctness."
            ),
            "coherence": (
                "- Measures the collective quality of all sentences in the summary.\n"
                "- Consider whether the summary is well-structured and ideas flow naturally from sentence to sentence.\n"
                "- Penalize summaries that are just a list of events without an overall coherent structure."
            ),
            "5w1h": (
                "- Measures maintenance of important information from the source document.\n"
                "- Consider whether the summary preserves the relevant who, what, when, where, why, and how.\n"
                "- Do not penalize omission of minor Ws if the main ones are preserved."
            ),
        }

        super().__init__(template_str, dimension_defs)

    def _normalize_dimension(self, dimension: str) -> str:
        if dimension not in self.DIMENSION_ALIASES:
            raise ValueError(
                f"Unknown dimension '{dimension}'. "
                f"Valid dimensions: {list(self.DIMENSION_ALIASES.keys())}"
            )
        return self.DIMENSION_ALIASES[dimension]

    def format(self, dimension, **kwargs):
        dimension = self._normalize_dimension(dimension)

        article = (
            kwargs.get("question")
            or kwargs.get("article")
            or kwargs.get("document")
            or kwargs.get("item")
        )
        summary = kwargs.get("response") or kwargs.get("summary")

        if article is None or summary is None:
            raise ValueError(
                "Both source text and summary must be provided to format the "
                "HitzBasseSummevalTemplate."
            )

        context = self.template_str.format(
            dimension=dimension,
            dimension_defs=self.dimension_defs[dimension],
            article=article,
            summary=summary,
        )

        messages = [
            {"role": "system", "content": "system_prompt"},
            {"role": "user", "content": context},
        ]
        return messages