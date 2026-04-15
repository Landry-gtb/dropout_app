
# ─────────────────────────────────────────────────────────
# CATALOGUE DES RECOMMANDATIONS
# Clé : (dimension, niveau)  →  (badge_html, message)
# ─────────────────────────────────────────────────────────

CATALOGUE = {

    # ── PSYCHOLOGIQUE ─────────────────────────────────────

    ('fatigue', 'léger'): (
        '🟡 Léger',
        'Fatigue mentale détectée à un niveau modéré. '
        'Encourager des pauses régulières et une hygiène de sommeil adaptée.'
    ),
    ('fatigue', 'modéré'): (
        '🟠 Modéré',
        'Fatigue mentale significative. '
        'Orientation vers un entretien avec le service de santé universitaire recommandée. '
        'Envisager un allègement temporaire de la charge de travail.'
    ),
    ('fatigue', 'sévère'): (
        '🔴 Urgent',
        'Fatigue mentale critique — facteur de décrochage majeur. '
        'Prise en charge psychologique immédiate conseillée. '
        'Signalement au référent pédagogique et au service de santé.'
    ),

    ('stress', 'léger'): (
        '🟡 Léger',
        'Niveau de stress légèrement élevé. '
        'Proposer des ressources en ligne de gestion du stress (techniques de respiration, mindfulness).'
    ),
    ('stress', 'modéré'): (
        '🟠 Modéré',
        'Stress académique notable. '
        'Inscription à un atelier de gestion du stress recommandée. '
        'Vérifier la répartition des évaluations sur le semestre.'
    ),
    ('stress', 'sévère'): (
        '🔴 Urgent',
        'Stress très élevé — risque d\'épuisement imminent. '
        'Consultation psychologique urgente conseillée. '
        'Étudier un aménagement d\'examens si nécessaire.'
    ),

    ('motivation', 'léger'): (
        '🟡 Léger',
        'Légère baisse de motivation détectée. '
        'Valoriser les réussites récentes de l\'étudiant lors du prochain entretien.'
    ),
    ('motivation', 'modéré'): (
        '🟠 Modéré',
        'Motivation en baisse sensible. '
        'Entretien individuel avec le conseiller pédagogique recommandé. '
        'Explorer un éventuel réorientation partielle ou des options de spécialisation.'
    ),
    ('motivation', 'sévère'): (
        '🔴 Urgent',
        'Démotivation profonde — risque de décrochage immédiat. '
        'Entretien d\'urgence avec le responsable de formation. '
        'Envisager un bilan d\'orientation et explorer des parcours alternatifs.'
    ),

    ('confiance', 'léger'): (
        '🟡 Léger',
        'Légère baisse de confiance en soi. '
        'Encourager la participation orale en cours et valoriser les acquis.'
    ),
    ('confiance', 'modéré'): (
        '🟠 Modéré',
        'Manque de confiance notable impactant les résultats. '
        'Mise en place d\'un tutorat par les pairs conseillée. '
        'Travail sur les stratégies d\'apprentissage.'
    ),
    ('confiance', 'sévère'): (
        '🔴 Urgent',
        'Très faible confiance en soi — syndrome de l\'imposteur possible. '
        'Accompagnement psychologique et soutien scolaire intensif recommandés. '
        'Suivi hebdomadaire par le référent pédagogique.'
    ),

    # ── ACADÉMIQUE ────────────────────────────────────────

    ('moyenne', 'léger'): (
        '🟡 Léger',
        'Moyenne générale légèrement insuffisante. '
        'Identifier les matières les plus problématiques et proposer des ressources ciblées.'
    ),
    ('moyenne', 'modéré'): (
        '🟠 Modéré',
        'Résultats académiques préoccupants. '
        'Mise en place d\'un tutorat académique et révision du plan de travail personnel recommandés.'
    ),
    ('moyenne', 'sévère'): (
        '🔴 Urgent',
        'Résultats critiques — risque d\'échec semestrel élevé. '
        'Plan de soutien individualisé urgent. '
        'Vérifier l\'éligibilité à des rattrapages ou aménagements pédagogiques.'
    ),

    ('absent', 'léger'): (
        '🟡 Léger',
        'Légères absences répétées. '
        'Rappel des règles de présence et discussion sur les causes lors du prochain contact.'
    ),
    ('absent', 'modéré'): (
        '🟠 Modéré',
        'Taux d\'absentéisme préoccupant. '
        'Entretien de suivi avec le référent pédagogique pour identifier les obstacles. '
        'Vérifier les contraintes de transport ou de santé.'
    ),
    ('absent', 'sévère'): (
        '🔴 Urgent',
        'Absentéisme critique — décrochage en cours probable. '
        'Contact direct avec l\'étudiant et sa famille si mineur. '
        'Évaluation des conditions de vie et orientation vers les services sociaux.'
    ),

    ('devoir', 'léger'): (
        '🟡 Léger',
        'Quelques retards dans les rendus. '
        'Proposer un outil de planification (agenda, application de suivi des tâches).'
    ),
    ('devoir', 'modéré'): (
        '🟠 Modéré',
        'Non-rendu régulier des travaux. '
        'Atelier de méthodologie du travail universitaire conseillé. '
        'Mettre en place des rappels et un suivi personnalisé des échéances.'
    ),
    ('devoir', 'sévère'): (
        '🔴 Urgent',
        'Travaux systématiquement non rendus — situation critique. '
        'Entretien d\'urgence pour comprendre les blocages. '
        'Envisager un contrat pédagogique avec objectifs hebdomadaires.'
    ),

    ('matieres', 'léger'): (
        '🟡 Léger',
        '1 à 2 matières en difficulté. '
        'Cibler un soutien disciplinaire spécifique sur ces matières.'
    ),
    ('matieres', 'modéré'): (
        '🟠 Modéré',
        'Plusieurs matières échouées. '
        'Révision du programme de travail et tutorat multi-matières recommandés.'
    ),
    ('matieres', 'sévère'): (
        '🔴 Urgent',
        'Échecs multiples — année académique compromise. '
        'Bilan pédagogique complet nécessaire. '
        'Étudier les options de redoublement partiel ou de réorientation.'
    ),

    # ── ORGANISATIONNEL ───────────────────────────────────

    ('procrastinat', 'léger'): (
        '🟡 Léger',
        'Légère tendance à la procrastination. '
        'Partager des techniques simples : méthode Pomodoro, to-do liste quotidienne.'
    ),
    ('procrastinat', 'modéré'): (
        '🟠 Modéré',
        'Procrastination régulière impactant les résultats. '
        'Atelier de gestion du temps et des priorités conseillé. '
        'Travailler sur les causes profondes (peur de l\'échec, perfectionnisme).'
    ),
    ('procrastinat', 'sévère'): (
        '🔴 Urgent',
        'Procrastination sévère — paralysie académique possible. '
        'Accompagnement psycho-pédagogique combinant gestion du temps et soutien motivationnel. '
        'Suivi hebdomadaire avec objectifs quotidiens formalisés.'
    ),

    ('heures', 'léger'): (
        '🟡 Léger',
        'Temps d\'étude personnel légèrement insuffisant. '
        'Rappeler l\'importance du travail régulier hors cours.'
    ),
    ('heures', 'modéré'): (
        '🟠 Modéré',
        'Investissement personnel en dehors des cours trop faible. '
        'Identifier les obstacles (travail salarié, contraintes familiales) '
        'et proposer des créneaux d\'étude guidée en bibliothèque.'
    ),
    ('heures', 'sévère'): (
        '🔴 Urgent',
        'Quasi-absence de travail personnel — incompatible avec la réussite. '
        'Diagnostic complet de la situation (surcharge extérieure, désengagement). '
        'Envisager un aménagement du rythme d\'études si justifié.'
    ),

    # ── SOCIO-ÉCONOMIQUE ──────────────────────────────────

    ('logement', 'léger'): (
        '🟡 Léger',
        'Conditions de logement légèrement défavorables. '
        'Informer sur les espaces de travail disponibles sur le campus.'
    ),
    ('logement', 'modéré'): (
        '🟠 Modéré',
        'Situation de logement impactant les études. '
        'Orientation vers le service social universitaire pour évaluation des aides disponibles.'
    ),
    ('logement', 'sévère'): (
        '🔴 Urgent',
        'Conditions de logement précaires — facteur de décrochage direct. '
        'Orientation urgente vers le CROUS et les services sociaux. '
        'Évaluation du droit à une aide d\'urgence.'
    ),

    ('soutien', 'léger'): (
        '🟡 Léger',
        'Soutien de l\'entourage légèrement insuffisant. '
        'Encourager l\'étudiant à rejoindre des groupes de travail ou associations étudiantes.'
    ),
    ('soutien', 'modéré'): (
        '🟠 Modéré',
        'Manque de soutien social notable. '
        'Proposer un parrainage par un étudiant tuteur ou un mentor. '
        'Valoriser les réseaux de soutien internes à l\'établissement.'
    ),
    ('soutien', 'sévère'): (
        '🔴 Urgent',
        'Isolement social important — facteur aggravant majeur. '
        'Accompagnement socio-éducatif recommandé. '
        'Orientation vers l\'assistante sociale et les dispositifs d\'entraide étudiante.'
    ),

    ('trajet', 'léger'): (
        '🟡 Léger',
        'Temps de trajet légèrement long. '
        'Informer sur les horaires aménagés ou le télétravail pédagogique si disponible.'
    ),
    ('trajet', 'modéré'): (
        '🟠 Modéré',
        'Trajet domicile-université contraignant. '
        'Explorer les options de logement étudiant proche du campus. '
        'Vérifier l\'éligibilité aux aides de transport.'
    ),
    ('trajet', 'sévère'): (
        '🔴 Urgent',
        'Temps de trajet excessif compromettant la présence et la concentration. '
        'Accompagnement pour trouver un logement plus proche. '
        'Étudier la possibilité de cours à distance partiels.'
    ),

    ('environnement', 'léger'): (
        '🟡 Léger',
        'Environnement de travail à domicile perfectible. '
        'Conseiller l\'utilisation des salles de travail et bibliothèques du campus.'
    ),
    ('environnement', 'modéré'): (
        '🟠 Modéré',
        'Environnement défavorable à la concentration. '
        'Orientation vers les espaces de travail universitaires. '
        'Évaluer si des équipements (ordinateur, bureau) font défaut.'
    ),
    ('environnement', 'sévère'): (
        '🔴 Urgent',
        'Conditions de travail à domicile très précaires. '
        'Signalement au service social pour évaluation des besoins matériels. '
        'Accès prioritaire aux ressources numériques du campus.'
    ),

    ('lms', 'léger'): (
        '🟡 Léger',
        'Connexion au LMS irrégulière. '
        'Rappeler l\'importance du suivi des ressources en ligne déposées par les enseignants.'
    ),
    ('lms', 'modéré'): (
        '🟠 Modéré',
        'Faible utilisation de la plateforme d\'apprentissage. '
        'Session de formation aux outils numériques pédagogiques conseillée. '
        'Vérifier l\'accès à un équipement informatique fiable.'
    ),
    ('lms', 'sévère'): (
        '🔴 Urgent',
        'Quasi-absence sur le LMS — déconnexion pédagogique totale. '
        'Diagnostic urgent : problème technique, d\'accès ou de désengagement ? '
        'Contact direct de l\'enseignant référent recommandé.'
    ),
}

# ─────────────────────────────────────────────────────────
# MAPPING : mots-clés → clé de dimension
# ─────────────────────────────────────────────────────────
KEYWORDS_TO_DIM = {
    'fatigue':       ['fatigue'],
    'stress':        ['stress'],
    'motivation':    ['motivé', 'motivation'],
    'confiance':     ['confiance'],
    'moyenne':       ['moyenne'],
    'absent':        ['absent'],
    'devoir':        ['devoir', 'délais'],
    'matieres':      ['matières', 'echoué', 'échoué'],
    'procrastinat':  ['procrastin'],
    'heures':        ['heures'],
    'logement':      ['logement'],
    'soutien':       ['soutien', 'entourage'],
    'trajet':        ['trajet'],
    'environnement': ['environnement', 'domicile'],
    'lms':           ['lms', 'connexion'],
}


def _shap_to_niveau(shap_val: float) -> str:
    """Convertit une SHAP value en niveau de sévérité."""
    abs_val = abs(shap_val)
    if abs_val > 0.15:
        return 'sévère'
    elif abs_val > 0.05:
        return 'modéré'
    else:
        return 'léger'


def _dim_from_feature(feat_lower: str) -> str | None:
    """Retourne la dimension associée à un nom de feature, ou None."""
    for dim, kws in KEYWORDS_TO_DIM.items():
        if any(kw in feat_lower for kw in kws):
            return dim
    return None


# ─────────────────────────────────────────────────────────
# FONCTION PRINCIPALE 
# ─────────────────────────────────────────────────────────
def generer_recommandations(sv, feature_names: list, max_recos: int = 4) -> list[dict]:
    """
    Génère des recommandations enrichies à 3 niveaux de sévérité.

    Paramètres
    ----------
    sv           : np.ndarray — vecteur 1D des SHAP values (classe positive)
    feature_names: list[str]  — noms de features (même ordre que sv)
    max_recos    : int        — nombre maximum de recommandations à retourner

    Retourne
    --------
    list[dict] avec les clés :
        - 'badge'   : str  — ex. '🔴 Urgent'
        - 'message' : str  — texte de la recommandation
        - 'niveau'  : str  — 'léger' | 'modéré' | 'sévère'
        - 'dim'     : str  — dimension concernée
    """
    import numpy as np

    sv = np.asarray(sv, dtype=float).flatten()

    # Trier les features par SHAP décroissant (impact positif sur le risque)
    sorted_pairs = sorted(
        zip(sv.tolist(), feature_names),
        key=lambda x: x[0],
        reverse=True
    )

    recos      = []
    dims_vus   = set()   

    for shap_val, feat in sorted_pairs:
        if shap_val <= 0:
            break         

        dim = _dim_from_feature(feat.lower())
        if dim is None or dim in dims_vus:
            continue

        niveau = _shap_to_niveau(shap_val)
        cle    = (dim, niveau)

        if cle in CATALOGUE:
            badge, message = CATALOGUE[cle]
            recos.append({
                'badge':   badge,
                'message': message,
                'niveau':  niveau,
                'dim':     dim,
            })
            dims_vus.add(dim)

        if len(recos) >= max_recos:
            break

    # Fallback si aucune recommandation trouvée
    if not recos:
        recos.append({
            'badge':   '🔵 Suivi',
            'message': 'Entretien de suivi global avec le conseiller pédagogique recommandé.',
            'niveau':  'léger',
            'dim':     'général',
        })

    return recos


# ─────────────────────────────────────────────────────────
# COULEURS HTML PAR NIVEAU 
# ─────────────────────────────────────────────────────────
COULEURS_NIVEAU = {
    'léger':  {'bg': 'linear-gradient(135deg,#fffbf0,#fff8e8)', 'border': '#c9a84c', 'badge': '#c9a84c'},
    'modéré': {'bg': 'linear-gradient(135deg,#fff5eb,#fff0e0)', 'border': '#e67e22', 'badge': '#e67e22'},
    'sévère': {'bg': 'linear-gradient(135deg,#fdf0ee,#fff0ee)', 'border': '#c0392b', 'badge': '#c0392b'},
    'général':{'bg': 'linear-gradient(135deg,#eef4ff,#f0f4ff)', 'border': '#2d6a9f', 'badge': '#2d6a9f'},
}
