import openai
import pandas as pd
from sklearn.metrics import accuracy_score
from tqdm import tqdm
from retry import retry
import time

import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Get the API key from the environment
api_key = os.getenv("OPENAI_API_KEY")

# Set your OpenAI API key
client = openai.OpenAI(api_key)

@retry(tries=3, delay=1, backoff=2, jitter=(0.5, 2.0))
def translate_text(text, model, target_language):
    prompt = f"Translate the following text to {target_language}: {text}. Output only the translation."
    translation = ""
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": prompt}
            ],
            temperature=0,
            seed=20
        )
        translation = response.choices[0].message.content.strip().replace("\n","")
    except:
        translation = "openai error"
        pass

    return translation


@retry(tries=3, delay=1, backoff=2, jitter=(0.5, 2.0))
def predict(dataset, model, text, language, method):
    if dataset=='depression_reddit':
        if language=='english':
            prompt = f"categorize the following text with 1 of the 4 depression severity levels (0: Minimum, 1: Mild, 2: Moderate, 3: Severe): {text}. please output only 0, 1, 2, or 3"
            add_shot = "Here are some examples. ""Had a bit of a rough week at work, but overall, I'm doing okay. I’ve been feeling a little down, but I think it's just from being tired. Planning to spend the weekend relaxing and recharging. I'm looking forward to catching up on some hobbies."": Minimum. ""Lately, I've been feeling a bit more down than usual. It’s like a grey cloud follows me around. I'm still functioning and getting things done, but I find it harder to enjoy the things I used to. I'm trying to push through it, but it's tough."": Mild. ""I’ve been finding it really hard to get out of bed lately. Everything feels pointless, and I can’t remember the last time I genuinely felt happy. I’ve started skipping social events because I just don’t have the energy. I’m trying to keep it together at work, but it’s getting harder every day."": Moderate. ""I don't see the point in anything anymore. I’ve completely isolated myself from everyone, and even getting out of bed feels impossible. I’m constantly thinking that things would be better if I just disappeared. I feel like I'm trapped in this dark place with no way out."": Severe"
        elif language=='turkish':
            prompt = f"Aşağıdaki metni 4 depresyon şiddet seviyesinden 1'i (0: Minimum, 1: Hafif, 2: Orta, 3: Şiddetli) ile kategorilendirin: {text}. lütfen yalnızca 0, 1, 2 veya 3 çıktısını verin"
            add_shot = "İşte birkaç örnek. ""İş yerinde biraz zor bir hafta geçirdim ama genel olarak iyiyim. Kendimi biraz kötü hissediyorum ama bunun sadece yorgun olmaktan kaynaklandığını düşünüyorum. Hafta sonunu dinlenerek ve enerji toplayarak geçirmeyi planlıyorum. Bazı hobilerimi yapmayı dört gözle bekliyorum."": Minimum. ""Son zamanlarda, normalden biraz daha kötü hissediyorum. Sanki etrafımda gri bir bulut dolaşıyor. Hala işimi yapıyorum ve işlerimi hallediyorum ama eskiden yaptığım şeylerden zevk almam zorlaşıyor. Bunu aşmaya çalışıyorum ama zor."": Hafif. ""Son zamanlarda yataktan çıkmakta gerçekten zorlanıyorum. Her şey anlamsız geliyor ve gerçekten mutlu hissettiğim son zamanı hatırlamıyorum. Sosyal etkinlikleri atlamaya başladım çünkü enerjim yok. İş yerinde kendimi toparlamaya çalışıyorum ama her geçen gün daha da zorlaşıyor."": Orta. ""Artık hiçbir şeyin anlamını göremiyorum. Kendimi herkesten tamamen soyutladım ve yataktan çıkmak bile imkansız gibi geliyor. Sürekli olarak ortadan kaybolursam her şeyin daha iyi olacağını düşünüyorum. Çıkış yolu olmayan karanlık bir yerde sıkışıp kalmış gibi hissediyorum."": Şiddetli"
        elif language=='portuguese':
            prompt = f"categorize o texto a seguir com 1 dos 4 níveis de gravidade da depressão (0: Mínimo, 1: Leve, 2: Moderado, 3: Grave): {text}. por favor, imprima apenas 0, 1, 2 ou 3"
            add_shot = "Aqui estão alguns exemplos. ""Tive uma semana difícil no trabalho, mas no geral estou bem. Estou a sentir-me um pouco desanimado, mas acho que é só por estar cansado. Planear passar o fim de semana a relaxar e a recarregar energias. Estou ansioso por pôr alguns passatempos em dia."": Mínimo. ""Ultimamente, tenho-me sentido um pouco mais deprimido do que o habitual. É como se uma nuvem cinzenta me seguisse. Ainda estou a trabalhar e a fazer as coisas, mas acho mais difícil desfrutar das coisas que costumava fazer. Estou a tentar ultrapassar isto, mas é difícil."": Leve. ""Tenho tido muita dificuldade em sair da cama ultimamente. Tudo parece inútil e não me consigo lembrar da última vez que me senti genuinamente feliz. Comecei a faltar a eventos sociais porque simplesmente não tenho energia. Estou a tentar controlar-me no trabalho, mas está a tornar-se mais difícil a cada dia que passa."": Moderado. ""Já não vejo sentido em nada. Isolei-me completamente de todos e até sair da cama parece impossível. Estou constantemente a pensar que as coisas seriam melhores se eu simplesmente desaparecesse. Sinto que estou preso neste lugar escuro e sem saída."": Grave"
        elif language=='german':
            prompt = f"Kategorisieren Sie den folgenden Text mit einem der vier Schweregrade einer Depression (0: minimal, 1: leicht, 2: mittelschwer, 3: schwer): {text}. bitte nur 0, 1, 2 oder 3 ausgeben"
            add_shot = "Hier sind einige Beispiele. „Hatten eine ziemlich harte Woche bei der Arbeit, aber insgesamt geht es mir gut. Ich fühle mich ein bisschen niedergeschlagen, aber ich glaube, das liegt einfach an der Müdigkeit. Ich habe vor, das Wochenende zu verbringen, um mich zu entspannen und neue Kraft zu tanken. Ich freue mich darauf, ein paar Hobbys wieder aufzunehmen.“: Minimal. „In letzter Zeit fühle ich mich ein bisschen niedergeschlagener als sonst. Es ist, als würde mir eine graue Wolke folgen. Ich funktioniere immer noch und erledige Dinge, aber es fällt mir schwerer, die Dinge zu genießen, die ich früher genossen habe. Ich versuche, mich durchzukämpfen, aber es ist hart.“: Leicht. „In letzter Zeit fällt es mir wirklich schwer, aus dem Bett zu kommen. Alles fühlt sich sinnlos an und ich kann mich nicht erinnern, wann ich das letzte Mal wirklich glücklich war. Ich habe angefangen, gesellschaftliche Veranstaltungen auszulassen, weil mir einfach die Energie fehlt. Ich versuche, mich bei der Arbeit zusammenzureißen, aber es wird von Tag zu Tag schwieriger.“: mittelschwer. „Ich sehe in nichts mehr den Sinn. Ich habe mich völlig von allen isoliert und es ist mir sogar unmöglich, aus dem Bett aufzustehen. Ich denke ständig, dass alles besser wäre, wenn ich einfach verschwinden würde. Ich fühle mich, als wäre ich an diesem dunklen Ort gefangen und hätte keinen Ausweg.“: Schwer"
        elif language=='finnish':
            prompt = f"Luokittele seuraava teksti yhdelle neljästä masennuksen vaikeusasteesta (0: minimi, 1: lievä, 2: kohtalainen, 3: vaikea): {text}. anna vain 0, 1, 2 tai 3"
            add_shot = "Tässä muutamia esimerkkejä. ""Töissä oli vähän rankka viikko, mutta kaiken kaikkiaan voin hyvin. Olen ollut hieman masentunut, mutta luulen sen johtuvan vain väsymyksestä. Viikonloppu on tarkoitus viettää rentoutuen ja latautuen. Odotan innolla harrastuksia."": Minimi. ""Olen viime aikoina tuntenut oloni hieman masemmaksi kuin tavallisesti. Tuntuu kuin harmaa pilvi seuraisi minua. Toimin edelleen ja teen asioita, mutta minun on vaikeampi nauttia asioista, joita ennen. Yritän päästä sen läpi, mutta se on vaikeaa."": Lievä. ""Minun on ollut viime aikoina todella vaikeaa nousta sängystä. Kaikki tuntuu turhalta, enkä muista, milloin viimeksi olisin aidosti ollut onnellinen. Olen alkanut jättää väliin sosiaaliset tapahtumat, koska minulla ei vain ole energiaa. Yritän pitää sen yhdessä töissä, mutta se on päivä päivältä vaikeampaa."": Kohtalainen. ""En näe järkeä enää missään. Olen täysin eristänyt itseni kaikista, ja jopa sängystä nouseminen tuntuu mahdottomalta. Ajattelen jatkuvasti, että asiat olisivat paremmin, jos vain katoaisin. Minusta tuntuu, että olen loukussa tähän pimeään paikkaan, josta ei ole ulospääsyä."": Vaikea"
        elif language=='greek':
            prompt = f"κατηγοριοποιήστε το παρακάτω κείμενο με 1 από τα 4 επίπεδα σοβαρότητας κατάθλιψης (0: Ελάχιστο, 1: Ήπιο, 2: Μέτριο, 3: Σοβαρό): {text}. παρακαλώ εξάγετε μόνο 0, 1, 2 ή 3"
            add_shot = "Εδώ είναι μερικά παραδείγματα. «Είχα μια δύσκολη εβδομάδα στη δουλειά, αλλά γενικά, τα πάω καλά. Νιώθω λίγο πεσμένος, αλλά νομίζω ότι είναι μόνο από κούραση. Σχεδιάζετε να περάσετε το Σαββατοκύριακο χαλαρώνοντας και επαναφορτίζοντας. Ανυπομονώ να προλάβω κάποια χόμπι."": Ελάχιστο. ""Τον τελευταίο καιρό αισθάνομαι λίγο πιο πεσμένος από ότι συνήθως. Είναι σαν ένα γκρίζο σύννεφο να με ακολουθεί. Εξακολουθώ να λειτουργώ και να κάνω τα πράγματα, αλλά δυσκολεύομαι να απολαμβάνω τα πράγματα που συνήθιζα. Προσπαθώ να το ξεπεράσω, αλλά είναι δύσκολο."": Ήπιο. ""Δυσκολεύομαι πολύ να σηκωθώ από το κρεβάτι τον τελευταίο καιρό. Όλα φαίνονται άσκοπα και δεν μπορώ να θυμηθώ την τελευταία φορά που ένιωσα αληθινά χαρούμενος. Έχω αρχίσει να παρακάμπτω τις κοινωνικές εκδηλώσεις γιατί απλά δεν έχω την ενέργεια. Προσπαθώ να το κρατήσω μαζί στη δουλειά, αλλά γίνεται όλο και πιο δύσκολο κάθε μέρα."": Μέτρια. ""Δεν βλέπω το νόημα σε τίποτα πια. Έχω απομονωθεί εντελώς από όλους, και ακόμη και το να σηκωθώ από το κρεβάτι μου φαίνεται αδύνατον. Σκέφτομαι συνεχώς ότι τα πράγματα θα ήταν καλύτερα αν απλώς εξαφανιζόμουν. Νιώθω σαν να είμαι παγιδευμένος σε αυτό το σκοτεινό μέρος χωρίς διέξοδο."": Σοβαρό"
        elif language=='french':
            prompt = f"catégorisez le texte suivant avec 1 des 4 niveaux de gravité de la dépression (0 : minimal, 1 : léger, 2 : modéré, 3 : sévère) : {text}. veuillez sortir uniquement 0, 1, 2 ou 3"
            add_shot = "Voici quelques exemples. « J'ai eu une semaine un peu difficile au travail, mais dans l'ensemble, je vais bien. Je me sens un peu déprimé, mais je pense que c'est juste à cause de la fatigue. Je prévois de passer le week-end à me détendre et à me ressourcer. J'ai hâte de rattraper certains passe-temps. » » : Minimum. « Ces derniers temps, je me sens un peu plus déprimé que d'habitude. C'est comme si un nuage gris me suivait partout. Je fonctionne toujours et j'accomplis des choses, mais j'ai plus de mal à profiter des choses que j'avais l'habitude de faire. J'essaie de m'en sortir, mais c'est dur. » » : Léger. « J'ai vraiment du mal à sortir du lit ces derniers temps. Tout semble inutile, et je ne me souviens pas de la dernière fois où je me suis senti vraiment heureux. J'ai commencé à éviter les événements sociaux parce que je n'ai tout simplement pas l'énergie. J'essaie de me ressaisir au travail, mais c'est de plus en plus difficile chaque jour. » » : Modéré. « Je ne vois plus l'intérêt de quoi que ce soit. Je me suis complètement isolée de tout le monde, et même sortir du lit me semble impossible. Je pense constamment que les choses iraient mieux si je disparaissais tout simplement. J'ai l'impression d'être piégée dans cet endroit sombre sans issue."": Grave"

    elif dataset=='depression_tweet':
        if language=='english':
            prompt = f"Does the following text indicate depression? Answer with 0 for No and 1 for Yes: {text}"
        elif language=='turkish':
            prompt = f"Aşağıdaki metin depresyona mı işaret ediyor? Hayır için 0, Evet için 1 ile cevaplayın: {text}"
        elif language=='portuguese':
            prompt = f"O texto a seguir indica depressão? Responda com 0 para Não e 1 para Sim: {text}"
        elif language=='german':
            prompt = f"Deutet der folgende Text auf eine Depression hin? Antworten Sie mit 0 für Nein und 1 für Ja: {text}"
        elif language=='finnish':
            prompt = f"Osoittaako seuraava teksti masennuksesta? Vastaa 0:lla ei ja 1 kyllä: {text}"

        # prompt = f"Does the following text indicate depression? Answer with 0 for No and 1 for Yes: {text}"
        # prompt = f"Le texte suivant indique-t-il une dépression ? Répondez avec 0 pour Non et 1 pour Oui: {text}. "
        # prompt = f"Υποδηλώνει κατάθλιψη το παρακάτω κείμενο; Απάντησε με 0 για Όχι και 1 για Ναι: {text}"
    elif dataset=='suicide':
        if language=='turkish':
            prompt = f"İçerik intihara yönelik değilse 0'a, içerik intihar düşüncesiyle neredeyse ilgiliyse 1'e ve içerik bir kullanıcının intihar etme potansiyeline işaret ettiğini gösteriyorsa 2'ye sınıflandırın: {text}"
        elif language=='french':
            prompt = f"classer à 0 si le contenu n'est pas suicidaire, à 1 si le contenu est presque lié à des idées suicidaires et à 2 si le contenu indique qu'un utilisateur est potentiellement suicidaire: {text}"
        elif language=='portuguese':
            prompt = f"categorizar em 0 se o conteúdo não for suicida, em 1 se o conteúdo estiver quase relacionado à ideação suicida e em 2 se o conteúdo indicar um usuário potencialmente cometendo suicídio: {text}"
        elif language=='german':
            prompt = f"Kategorisierung auf 0, wenn der Inhalt nicht selbstmörderisch ist, auf 1, wenn der Inhalt nahe an Selbstmordgedanken liegt, und auf 2, wenn der Inhalt darauf hinweist, dass ein Benutzer möglicherweise Selbstmord begeht: {text}"
        elif language=='greek':
            prompt = f"κατηγοριοποιήστε σε 0 εάν το περιεχόμενο δεν είναι αυτοκτονικό, σε 1 εάν το περιεχόμενο σχετίζεται σχεδόν με αυτοκτονικό ιδεασμό και σε 2 εάν το περιεχόμενο υποδεικνύει έναν χρήστη που ενδέχεται να αυτοκτονήσει: {text}"
        elif language=='finnish':
            prompt = f"luokittele arvoon 0, jos sisältö ei ole itsemurhaa, 1:een, jos sisältö liittyy melkein itsemurha-ajatukseen, ja 2:een, jos sisältö viittaa käyttäjän mahdollisesti tekemiseen itsemurhaan: {text}"

    if method=="add_shot":
        prompt += add_shot

    prediction = ""
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": prompt}
            ],
            temperature=0,
            seed=20
        )
        prediction = response.choices[0].message.content.strip().replace("\n","")
    except:
        prediction = "openai error"
        pass
    return prediction


# dataset = 'suicide'
method = "add_shot"
dataset = 'depression_reddit'
model = 'gpt-3.5-turbo'
# model = 'gpt-4o-mini'
languages = ['english']
# languages = ['turkish', 'french', 'portuguese', 'german', 'finnish', 'greek']
# languages = ['french', 'portuguese', 'german']
# languages = ['finnish', 'greek']

# Read data from CSV file
if dataset=='depression_reddit':
    df = pd.read_csv("data/Depression_Severity_Dataset-main/Reddit_depression_dataset.csv", quotechar='"')
elif dataset=='depression_tweet':
    df = pd.read_csv("data/depression_tweet/test.csv")
elif dataset=='suicide':
    df = pd.read_csv("data/suicide/Labelled_tweets.tsv", header=0, delimiter="\t", quoting=3)
    df = df.rename(columns={'tweet': 'text'})

# all_results_file = open('results/all_results.txt','w')

for language in languages:
    print(language)
    # Translate texts and predict depression symptoms
    translated_texts = []
    predictions = []
    if dataset=='depression_reddit':
        if language!="english":
            f_trans=open('results/'+method+'_'+model+'_reddit_translations_'+language+'.txt', 'w')

        f_pred=open('results/'+method+'_'+model+'_reddit_preds_'+language+'.txt', 'w')
    if dataset=='depression':
        f_trans=open('results/'+model+'_dep_tweet_translations_'+language+'.txt', 'w')
        f_pred=open('results/'+model+'_dep_tweet_preds_'+language+'.txt', 'w')
    elif dataset=='suicide':
        f_trans=open('results/'+model+'_suic_translations_'+language+'.txt', 'w')
        f_pred=open('results/'+model+'_suic_preds_'+language+'.txt', 'w')

    for index, row in tqdm(df.iterrows(), total=len(df), desc="Processing rows"):
        translated_text = row['text'].strip()
        if language!="english":
            translated_text = translate_text(row['text'].strip(), model, language)
            translated_texts.append(translated_text)
            f_trans.write(str(index)+": "+translated_text.replace("\n", " ").replace("\r", " ")+"\n")

        prediction = predict(dataset, model, translated_text, language, method)
        predictions.append(prediction)
        f_pred.write(str(index)+": "+str(prediction.replace("\n", " ").replace("\r", " "))+"\n")

    if language!="english":
        f_trans.close()
    f_pred.close()

    # Compare predictions with ground truth labels
    # df['translated_text'] = translated_texts
    df['prediction'] = predictions

    # Save results to a new CSV file
    df.to_csv('results/'+method+'_'+model+'_'+dataset+'_predictions_'+language+'.csv', index=False)
    # df.to_csv('results/translated_texts_and_predictions_'+language+'.csv', index=False)

    # accuracy = accuracy_score(df['label'], df['prediction'])

# Print the results
# print("Accuracy:", accuracy)
#
# f = open('results/dep_tweet.txt','w')
# f.write("Run on test\n\n")
# f.write("Language: "+language+"\n")
# f.write("Model: " +model+"\n")
# f.write("Accuracy: " + str(accuracy)+"\n")
# f.close()
