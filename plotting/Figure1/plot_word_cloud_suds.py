from wordcloud import WordCloud
import matplotlib.pyplot as plt
import numpy as np

phrases = {
    "sub004": [
        "Trigger anticipation", "Flip light switch wrong way",
        "Walking through door without tapping", "Put books on desk in wrong order",
        "Rearranging items in wrong order"
    ],
    "sub005": [
        "Touch orange juice", "Removing sheet from bed", "Put gloves from bed in bag",
        "Vacuum bed", "Moving feet on dirty floor", "Opening fridge wrong way",
        "Contaminate tablet", "Wash hands incorrect", "Looking at cluttered room",
        "Lying down in bed after being in living room", "Looking at a wart"
    ],
    "sub007": [
        "Scrolling through YouTube App", "Writing card to husband", "Touching floor",
        "Touching bathroom sink", "Taking remote control to bathroom",
        "Touching dirty clothes", "Being alone in living room", "Being alone outside for 10 sec",
        "Walking alone on the street", "Getting mail from mailbox", "Sending dm to mom",
        "Wrapping gift for husband", "Touching toothbrush"
    ],
    "sub009": [
        "Going through door incorrectly", "Plug in electric toothbrush",
        "Writing an e-mail", "Watch image of a knife",
        "Watch video of a car accident", "Turning oven on and off"
    ],
    "sub010": [
        "View picture of young girl", "Read about sexual thoughts",
        "Read news article about children", "Write text about harm", "View picture of bikini"
    ],
    "sub011": [
        "View ad including 9.99", "View ad including word holy",
        "View word Anti",
        "Text email including 666", "View number 666", "Write number 6 on paper",
    ],
    "sub012": [
        "Touching floor", "Petting cat", "Touching trash can", "Touching objects in garage",
        "Contaminate fridge", "Touch objects in kitchen", "Look at image of urine" 
    ],
}

# Colors for each subject
subjects = list(phrases.keys())
colors = plt.cm.tab10(np.linspace(0, 1, len(subjects)))
color_map = {sub: tuple((c[:3] * np.array([255,255,255])).astype(int)) for sub, c in zip(subjects, colors)}

# Flatten phrases into one string per subject
text_data = {}
for sub, items in phrases.items():
    text_data[sub] = " ".join(items)

# Combine into one dict for wordcloud
all_phrases = {phrase: 1 for sublist in phrases.values() for phrase in sublist}  # freq = 1 for all

# Custom color function
def color_func(word, font_size, position, orientation, random_state=None, **kwargs):
    for sub, items in phrases.items():
        if word in items:
            return "rgb{}".format(color_map[sub])
    return "black"

wc = WordCloud(
    width=1000,
    height=600,
    background_color="white",
    prefer_horizontal=0.7,
    min_font_size=10,
    max_font_size=40,
    random_state=42
).generate_from_frequencies(all_phrases)

wordcloud_svg = wc.to_svg(embed_font=True)
f = open("figures/Figure1/fig1_wordcloud_SUDS_triggers.svg","w+")
f.write(wordcloud_svg )
f.close()
