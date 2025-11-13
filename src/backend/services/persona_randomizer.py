"""
Persona Randomizer Service

Generates random persona configurations for quick testing and creative exploration.
"""

import random
from typing import Dict, List, Any, Optional
from backend.models.persona import ContentRating


class PersonaRandomizer:
    """
    Service for generating randomized persona configurations.
    
    Provides creative combinations of appearance, personality, themes, and styles
    for rapid persona creation and experimentation.
    """
    
    # Appearance components
    GENDERS = ["female", "male", "non-binary"]
    
    AGE_RANGES = [
        "early 20s", "mid 20s", "late 20s",
        "early 30s", "mid 30s", "late 30s",
        "early 40s", "mid 40s"
    ]
    
    BODY_TYPES = [
        "athletic", "slim", "curvy", "muscular", "average",
        "petite", "plus-size", "fit", "toned"
    ]
    
    HAIR_COLORS = [
        "blonde", "brown", "black", "red", "auburn",
        "platinum blonde", "dark brown", "light brown",
        "jet black", "chestnut", "honey blonde"
    ]
    
    HAIR_STYLES = [
        "long flowing", "shoulder-length", "short pixie cut",
        "wavy", "straight", "curly", "bob cut", "layered",
        "braided", "ponytail", "messy bun", "sleek"
    ]
    
    EYE_COLORS = [
        "brown", "blue", "green", "hazel", "amber",
        "gray", "dark brown", "light blue", "emerald green"
    ]
    
    SKIN_TONES = [
        "fair", "light", "medium", "olive", "tan",
        "bronze", "dark", "deep", "golden"
    ]
    
    ETHNICITIES = [
        "Caucasian", "Asian", "African", "Hispanic", "Middle Eastern",
        "Mixed", "Mediterranean", "Scandinavian", "Latin American"
    ]
    
    CLOTHING_STYLES = [
        "casual streetwear", "professional business attire",
        "athletic activewear", "trendy fashion", "elegant formal wear",
        "bohemian chic", "minimalist modern", "vintage style",
        "urban casual", "smart casual"
    ]
    
    ACCESSORIES = [
        "stylish glasses", "designer sunglasses", "statement jewelry",
        "minimal jewelry", "watch", "earrings", "necklace",
        "bracelet", "no accessories"
    ]
    
    # Personality traits
    PERSONALITY_TYPES = [
        "outgoing and energetic", "calm and thoughtful", "creative and artistic",
        "analytical and logical", "empathetic and caring", "confident and bold",
        "friendly and approachable", "mysterious and intriguing",
        "humorous and witty", "serious and professional", "adventurous and daring",
        "nurturing and supportive", "ambitious and driven", "laid-back and easygoing"
    ]
    
    COMMUNICATION_STYLES = [
        "warm and conversational", "direct and clear", "poetic and expressive",
        "casual and friendly", "professional and polished", "playful and fun",
        "intellectual and articulate", "down-to-earth", "inspirational"
    ]
    
    INTERESTS = [
        "technology and innovation", "fashion and beauty", "fitness and wellness",
        "travel and adventure", "food and cooking", "art and design",
        "music and entertainment", "business and entrepreneurship",
        "science and education", "gaming and esports", "photography",
        "writing and literature", "environmental sustainability",
        "social causes and activism", "personal development"
    ]
    
    # Content themes by category
    CONTENT_THEME_CATEGORIES = {
        "fitness": [
            "workout routines", "nutrition tips", "fitness motivation",
            "gym lifestyle", "home workouts", "strength training"
        ],
        "fashion": [
            "outfit ideas", "style tips", "fashion trends",
            "wardrobe essentials", "seasonal fashion", "accessory styling"
        ],
        "tech": [
            "tech reviews", "software tutorials", "gadget unboxing",
            "coding tips", "AI and machine learning", "app recommendations"
        ],
        "lifestyle": [
            "daily routines", "productivity hacks", "self-care",
            "home decor", "life advice", "mindfulness"
        ],
        "food": [
            "recipe sharing", "cooking tips", "restaurant reviews",
            "meal prep", "healthy eating", "food photography"
        ],
        "travel": [
            "travel guides", "destination recommendations", "travel tips",
            "adventure stories", "budget travel", "luxury travel"
        ],
        "business": [
            "entrepreneurship", "marketing strategies", "business tips",
            "side hustles", "career advice", "financial freedom"
        ],
        "creative": [
            "art tutorials", "creative process", "design inspiration",
            "photography tips", "writing advice", "music creation"
        ],
        "gaming": [
            "game reviews", "gaming tips", "esports", "streaming",
            "game development", "gaming culture"
        ],
        "wellness": [
            "mental health", "meditation", "wellness routines",
            "holistic health", "stress management", "personal growth"
        ]
    }
    
    # Style preferences
    AESTHETICS = [
        "modern minimalist", "vibrant and colorful", "dark and moody",
        "bright and airy", "warm and cozy", "sleek and professional",
        "artistic and creative", "natural and organic", "bold and edgy",
        "soft and dreamy", "industrial chic", "vintage retro"
    ]
    
    PHOTOGRAPHY_STYLES = [
        "high contrast", "soft lighting", "dramatic shadows",
        "natural light", "studio quality", "candid moments",
        "cinematic", "documentary style", "editorial fashion"
    ]
    
    COLOR_PALETTES = [
        "warm tones", "cool tones", "neutral palette", "vibrant colors",
        "pastel shades", "monochromatic", "earth tones", "neon accents"
    ]
    
    # Platform focuses
    PLATFORMS = [
        ["instagram", "tiktok"],
        ["instagram", "youtube"],
        ["tiktok", "youtube"],
        ["instagram", "twitter"],
        ["onlyfans", "instagram"],
        ["youtube", "twitch"],
        ["instagram", "pinterest"]
    ]
    
    @classmethod
    def generate_random_name(cls) -> str:
        """Generate a random persona name."""
        first_names = [
            "Emma", "Olivia", "Ava", "Sophia", "Isabella",
            "Mia", "Charlotte", "Amelia", "Harper", "Evelyn",
            "Luna", "Aria", "Zoe", "Stella", "Maya",
            "Alex", "Jordan", "Riley", "Casey", "Morgan",
            "Ethan", "Liam", "Noah", "Oliver", "James",
            "Lucas", "Mason", "Logan", "Alexander", "Elijah"
        ]
        
        last_names = [
            "Smith", "Johnson", "Williams", "Brown", "Jones",
            "Garcia", "Miller", "Davis", "Rodriguez", "Martinez",
            "Anderson", "Taylor", "Thomas", "Moore", "Jackson",
            "Lee", "Walker", "Hall", "Young", "King"
        ]
        
        # Sometimes use nicknames or single names
        if random.random() < 0.3:
            nicknames = [
                "The Artist", "The Creator", "The Explorer", "The Dreamer",
                "The Innovator", "The Wanderer", "The Visionary",
                "Soul", "Vibe", "Essence", "Spirit", "Phoenix"
            ]
            return f"{random.choice(first_names)} {random.choice(nicknames)}"
        
        return f"{random.choice(first_names)} {random.choice(last_names)}"
    
    @classmethod
    def generate_random_appearance(cls, detailed: bool = True) -> str:
        """
        Generate a random appearance description.
        
        Args:
            detailed: If True, generates more detailed description
            
        Returns:
            Appearance description string
        """
        gender = random.choice(cls.GENDERS)
        age = random.choice(cls.AGE_RANGES)
        body_type = random.choice(cls.BODY_TYPES)
        hair_color = random.choice(cls.HAIR_COLORS)
        hair_style = random.choice(cls.HAIR_STYLES)
        eye_color = random.choice(cls.EYE_COLORS)
        skin_tone = random.choice(cls.SKIN_TONES)
        ethnicity = random.choice(cls.ETHNICITIES)
        clothing = random.choice(cls.CLOTHING_STYLES)
        accessories = random.choice(cls.ACCESSORIES)
        
        if detailed:
            parts = [
                f"A {age} {gender} person with a {body_type} build.",
                f"{ethnicity} with {skin_tone} skin and {eye_color} eyes.",
                f"Has {hair_style} {hair_color} hair.",
                f"Typically wears {clothing}",
            ]
            
            if accessories != "no accessories":
                parts.append(f"with {accessories}.")
            else:
                parts[-1] += "."
                
            parts.extend([
                "Professional appearance with confident posture.",
                "Photogenic with expressive facial features.",
                "Natural, authentic look suitable for social media content."
            ])
            
            return " ".join(parts)
        else:
            return f"{age} {gender}, {body_type} {ethnicity}, {hair_style} {hair_color} hair, {eye_color} eyes, {skin_tone} skin"
    
    @classmethod
    def generate_random_personality(cls, detailed: bool = True) -> str:
        """
        Generate a random personality description.
        
        Args:
            detailed: If True, generates more detailed description
            
        Returns:
            Personality description string
        """
        personality_type = random.choice(cls.PERSONALITY_TYPES)
        communication = random.choice(cls.COMMUNICATION_STYLES)
        interests = random.sample(cls.INTERESTS, k=random.randint(2, 4))
        
        if detailed:
            parts = [
                f"Personality is {personality_type}.",
                f"Communication style is {communication}.",
                f"Passionate about {', '.join(interests[:-1])} and {interests[-1]}.",
            ]
            
            # Add some random personality quirks
            quirks = [
                "Known for authentic and relatable content.",
                "Engages deeply with followers and community.",
                "Values creativity and self-expression.",
                "Believes in continuous learning and growth.",
                "Inspires others through positive energy.",
                "Maintains work-life balance and mental wellness.",
                "Champions inclusivity and diversity."
            ]
            parts.append(random.choice(quirks))
            
            return " ".join(parts)
        else:
            return f"{personality_type}, {communication}"
    
    @classmethod
    def generate_random_themes(cls, count: Optional[int] = None) -> List[str]:
        """
        Generate random content themes.
        
        Args:
            count: Number of themes to generate (default: random 3-6)
            
        Returns:
            List of content theme strings
        """
        if count is None:
            count = random.randint(3, 6)
        
        # Pick 1-2 categories
        num_categories = random.randint(1, 2)
        categories = random.sample(list(cls.CONTENT_THEME_CATEGORIES.keys()), k=num_categories)
        
        themes = []
        for category in categories:
            category_themes = cls.CONTENT_THEME_CATEGORIES[category]
            num_from_category = min(count - len(themes), len(category_themes))
            themes.extend(random.sample(category_themes, k=random.randint(1, num_from_category)))
        
        # Fill remaining with random themes from any category
        while len(themes) < count:
            category = random.choice(list(cls.CONTENT_THEME_CATEGORIES.keys()))
            theme = random.choice(cls.CONTENT_THEME_CATEGORIES[category])
            if theme not in themes:
                themes.append(theme)
        
        return themes[:count]
    
    @classmethod
    def generate_random_style_preferences(cls) -> Dict[str, str]:
        """
        Generate random style preferences.
        
        Returns:
            Dictionary of style preference key-value pairs
        """
        return {
            "aesthetic": random.choice(cls.AESTHETICS),
            "photography_style": random.choice(cls.PHOTOGRAPHY_STYLES),
            "color_palette": random.choice(cls.COLOR_PALETTES),
            "platform_focus": ", ".join(random.choice(cls.PLATFORMS)),
            "content_frequency": random.choice(["daily", "2-3 times per week", "weekly"]),
            "post_time": random.choice(["morning", "afternoon", "evening", "night", "varies"])
        }
    
    @classmethod
    def generate_random_content_rating(cls) -> tuple[ContentRating, List[ContentRating]]:
        """
        Generate random content rating configuration.
        
        Returns:
            Tuple of (default_rating, allowed_ratings)
        """
        # Weight towards SFW content
        weights = [0.7, 0.2, 0.1]  # SFW, Moderate, NSFW
        default = random.choices(
            [ContentRating.SFW, ContentRating.MODERATE, ContentRating.NSFW],
            weights=weights
        )[0]
        
        # Allowed ratings always include default
        allowed = [default]
        
        # Optionally add other ratings
        if default == ContentRating.SFW:
            if random.random() < 0.3:
                allowed.append(ContentRating.MODERATE)
        elif default == ContentRating.MODERATE:
            if random.random() < 0.5:
                allowed.append(ContentRating.SFW)
            if random.random() < 0.2:
                allowed.append(ContentRating.NSFW)
        else:  # NSFW
            if random.random() < 0.4:
                allowed.extend([ContentRating.SFW, ContentRating.MODERATE])
        
        return default, allowed
    
    @classmethod
    def generate_random_platform_restrictions(cls) -> Dict[str, str]:
        """
        Generate random platform-specific restrictions.
        
        Returns:
            Dictionary of platform restrictions
        """
        restrictions = {}
        
        # Common platforms
        common_platforms = ["instagram", "tiktok", "twitter", "facebook"]
        
        # Randomly add restrictions for some platforms
        num_restrictions = random.randint(0, 3)
        platforms = random.sample(common_platforms, k=min(num_restrictions, len(common_platforms)))
        
        restriction_options = ["sfw_only", "moderate_allowed", "both"]
        
        for platform in platforms:
            # Instagram and Facebook typically more restricted
            if platform in ["instagram", "facebook"]:
                restrictions[platform] = random.choice(["sfw_only", "moderate_allowed"])
            else:
                restrictions[platform] = random.choice(restriction_options)
        
        return restrictions
    
    @classmethod
    def generate_complete_random_persona(
        cls,
        name: Optional[str] = None,
        detailed: bool = True
    ) -> Dict[str, Any]:
        """
        Generate a complete random persona configuration.
        
        Args:
            name: Optional custom name (generated if not provided)
            detailed: If True, generates more detailed descriptions
            
        Returns:
            Dictionary with complete persona configuration
        """
        default_rating, allowed_ratings = cls.generate_random_content_rating()
        
        return {
            "name": name or cls.generate_random_name(),
            "appearance": cls.generate_random_appearance(detailed=detailed),
            "personality": cls.generate_random_personality(detailed=detailed),
            "content_themes": cls.generate_random_themes(),
            "style_preferences": cls.generate_random_style_preferences(),
            "default_content_rating": default_rating,
            "allowed_content_ratings": allowed_ratings,
            "platform_restrictions": cls.generate_random_platform_restrictions(),
            "is_active": True
        }
    
    @classmethod
    def generate_random_batch(cls, count: int = 5) -> List[Dict[str, Any]]:
        """
        Generate multiple random personas.
        
        Args:
            count: Number of personas to generate
            
        Returns:
            List of persona configuration dictionaries
        """
        return [cls.generate_complete_random_persona() for _ in range(count)]
