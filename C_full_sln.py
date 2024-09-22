# For this task, we are using Seq2Seq model, which converts one sequence of text to other.
# Suitable for translation and summarization

# The model is 3.13GB large.

from transformers import TFAutoModelForSeq2SeqLM, AutoTokenizer
from datasets import load_dataset

# loading the Dailymail dataset

dataset = load_dataset('abisee/cnn_dailymail', '3.0.0')

# loading the tokenizer and model

tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-large")
model = TFAutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-large")

# input fpr few shot-learning

articles = [

{
    "Article" : """LONDON, England (Reuters) -- Harry Potter star Daniel Radcliffe gains access to a reported £20 million ($41.1 million) fortune as he turns 18 on Monday, \
    but he insists the money won't cast a spell on him. Daniel Radcliffe as Harry Potter in "Harry Potter and the Order of the Phoenix" To the disappointment of gossip \
    columnists around the world, the young actor says he has no plans to fritter his cash away on fast cars, drink and celebrity parties. "I don't plan to be one of those \
    people who, as soon as they turn 18, suddenly buy themselves a massive sports car collection or something similar," he told an Australian interviewer earlier this month. \
    "I don't think I'll be particularly extravagant. "The things I like buying are things that cost about 10 pounds -- books and CDs and DVDs." At 18, Radcliffe will be able \
    to gamble in a casino, buy a drink in a pub or see the horror film "Hostel: Part II," currently six places below his number one movie on the UK box office chart. Details \
    of how he'll mark his landmark birthday are under wraps. His agent and publicist had no comment on his plans. "I'll definitely have some sort of party," he said in an interview. \
    "Hopefully none of you will be reading about it." Radcliffe's earnings from the first five Potter films have been held in a trust fund which he has not been able to touch. \
    Despite his growing fame and riches, the actor says he is keeping his feet firmly on the ground. "People are always looking to say 'kid star goes off the rails,'" he told reporters \
    last month. "But I try very hard not to go that way because it would be too easy for them." His latest outing as the boy wizard in "Harry Potter and the Order of the Phoenix" is \
    breaking records on both sides of the Atlantic and he will reprise the role in the last two films.  Watch I-Reporter give her review of Potter's latest » . There is life beyond \
    Potter, however. The Londoner has filmed a TV movie called "My Boy Jack," about author Rudyard Kipling and his son, due for release later this year. He will also appear in "December \
    Boys," an Australian film about four boys who escape an orphanage. Earlier this year, he made his stage debut playing a tortured teenager in Peter Shaffer's "Equus." Meanwhile, he is \
    braced for even closer media scrutiny now that he's legally an adult: "I just think I'm going to be more sort of fair game," he told Reuters. E-mail to a friend . Copyright 2007 Reuters. \
    All rights reserved.This material may not be published, broadcast, rewritten, or redistributed.
""",

    "Translation": """LONDON, England (Reuters) -- Der „Harry Potter“-Star Daniel Radcliffe erhält angeblich ein Vermögen von 20 Millionen Pfund (41,1 Millionen Dollar), wenn er am Montag 18 \
    wird, aber er besteht darauf, dass das Geld ihn nicht verzaubern wird. Daniel Radcliffe als Harry Potter in „Harry Potter und der Orden des Phönix“ Zur Enttäuschung von Klatschkolumnisten \
    auf der ganzen Welt sagt der junge Schauspieler, er habe nicht vor, sein Geld für schnelle Autos, Alkohol und Promi-Partys zu verprassen. „Ich habe nicht vor, einer dieser Leute zu sein, \
    die sich gleich mit 18 eine riesige Sportwagensammlung oder etwas Ähnliches kaufen“, sagte er Anfang des Monats einem australischen Interviewer. „Ich glaube nicht, dass ich besonders extravagant \
    sein werde. Ich kaufe gern Dinge, die etwa 10 Pfund kosten – Bücher, CDs und DVDs.“ Mit 18 Jahren wird Radcliffe in der Lage sein, in einem Casino zu spielen, sich in einem Pub ein Getränk zu \
    kaufen oder den Horrorfilm „Hostel: Part II“ zu sehen, der derzeit sechs Plätze hinter seinem Nummer-1-Film in den britischen Kinocharts liegt. Einzelheiten darüber, wie er seinen runden Geburtstag \
    feiern wird, sind geheim. Sein Agent und sein Presseagent äußerten sich nicht zu seinen Plänen. „Ich werde auf jeden Fall eine Art Party veranstalten“, sagte er in einem Interview. „Hoffentlich \
    liest keiner von Ihnen davon.“ Radcliffes Einnahmen aus den ersten fünf Potter-Filmen wurden in einem Treuhandfonds gehalten, den er nicht anrühren konnte. Trotz seines wachsenden Ruhms und Reichtums \
    sagt der Schauspieler, dass er mit beiden Beinen fest auf dem Boden bleibt. „Die Leute wollen immer sagen, dass Kinderstars aus der Bahn geraten“, sagte er Reportern letzten Monat. „Aber ich \
    versuche sehr, das nicht zu tun, denn das würde zu einfach für sie sein." Sein jüngster Auftritt als Zaubererjunge in "Harry Potter und der Orden des Phönix" bricht auf beiden Seiten des Atlantiks \
    Rekorde und er wird die Rolle in den letzten beiden Filmen wieder spielen. Sehen Sie sich die Kritik von I-Reporter zu Potters neuestem Film an » . Es gibt jedoch ein Leben jenseits von Potter. \
    Der Londoner hat einen Fernsehfilm namens "My Boy Jack" über den Autor Rudyard Kipling und seinen Sohn gedreht, der noch in diesem Jahr in die Kinos kommen soll. Er wird auch in "December Boys" \
    auftreten, einem australischen Film über vier Jungen, die aus einem Waisenhaus fliehen. Anfang des Jahres gab er sein Bühnendebüt als gequälter Teenager in Peter Shaffers "Equus". In der Zwischenzeit \
    muss er sich auf noch genauere Medienbeobachtung einstellen, da er nun volljährig ist: "Ich denke, ich werde eher Freiwild sein", sagte er gegenüber Reuters. E-Mail an einen Freund . Copyright 2007 \
    Reuters. Alle Rechte vorbehalten. Dieses Material darf nicht veröffentlicht, gesendet, neu geschrieben oder weiterverteilt werden.
"""
},

{    
    "Article" : """In our Behind the Scenes series, CNN correspondents share their experiences in covering news and analyze the stories behind the events. Here, Soledad O'Brien takes users inside a \
    jail where many of the inmates are mentally ill. An inmate housed on the "forgotten floor," where many mentally ill inmates are housed in Miami before trial. MIAMI, Florida (CNN) -- The ninth floor \
    of the Miami-Dade pretrial detention facility is dubbed the "forgotten floor." Here, inmates with the most severe mental illnesses are incarcerated until they're ready to appear in court. Most often, \
    they face drug charges or charges of assaulting an officer --charges that Judge Steven Leifman says are usually "avoidable felonies." He says the arrests often result from confrontations with police. \
    Mentally ill people often won't do what they're told when police arrive on the scene -- confrontation seems to exacerbate their illness and they become more paranoid, delusional, and less likely to \
    follow directions, according to Leifman. So, they end up on the ninth floor severely mentally disturbed, but not getting any real help because they're in jail. We toured the jail with Leifman. He is \
    well known in Miami as an advocate for justice and the mentally ill. Even though we were not exactly welcomed with open arms by the guards, we were given permission to shoot videotape and tour the floor.  \
    Go inside the 'forgotten floor' » . At first, it's hard to determine where the people are. The prisoners are wearing sleeveless robes. Imagine cutting holes for arms and feet in a heavy wool sleeping \
    bag -- that's kind of what they look like. They're designed to keep the mentally ill patients from injuring themselves. That's also why they have no shoes, laces or mattresses. Leifman says about one-third \
    of all people in Miami-Dade county jails are mentally ill. So, he says, the sheer volume is overwhelming the system, and the result is what we see on the ninth floor. Of course, it is a jail, so it's not \
    supposed to be warm and comforting, but the lights glare, the cells are tiny and it's loud. We see two, sometimes three men -- sometimes in the robes, sometimes naked, lying or sitting in their cells. \
    "I am the son of the president. You need to get me out of here!" one man shouts at me. He is absolutely serious, convinced that help is on the way -- if only he could reach the White House. Leifman tells \
    me that these prisoner-patients will often circulate through the system, occasionally stabilizing in a mental hospital, only to return to jail to face their charges. It's brutally unjust, in his mind, and \
    he has become a strong advocate for changing things in Miami. Over a meal later, we talk about how things got this way for mental patients. Leifman says 200 years ago people were considered "lunatics" and \
    they were locked up in jails even if they had no charges against them. They were just considered unfit to be in society. Over the years, he says, there was some public outcry, and the mentally ill were moved \
    out of jails and into hospitals. But Leifman says many of these mental hospitals were so horrible they were shut down. Where did the patients go? Nowhere. The streets. They became, in many cases, the homeless, \
    he says. They never got treatment. Leifman says in 1955 there were more than half a million people in state mental hospitals, and today that number has been reduced 90 percent, and 40,000 to 50,000 people are \
    in mental hospitals. The judge says he's working to change this. Starting in 2008, many inmates who would otherwise have been brought to the "forgotten floor"  will instead be sent to a new mental health \
    facility -- the first step on a journey toward long-term treatment, not just punishment. Leifman says it's not the complete answer, but it's a start. Leifman says the best part is that it's a win-win solution. \
    The patients win, the families are relieved, and the state saves money by simply not cycling these prisoners through again and again. And, for Leifman, justice is served. E-mail to a friend.
""",

    "Translation": """In unserer Serie „Hinter den Kulissen“ berichten CNN-Korrespondenten über ihre Erfahrungen bei der Berichterstattung und analysieren die Hintergründe der Ereignisse. Hier führt Soledad \
    O'Brien die Benutzer in ein Gefängnis, in dem viele psychisch Kranke leben. Ein Häftling, der auf der „vergessenen Etage“ untergebracht ist, wo viele psychisch Kranke bis zu ihrem Prozess in Miami \
    untergebracht sind. MIAMI, Florida (CNN) – Die neunte Etage des Untersuchungsgefängnisses Miami-Dade wird als „vergessene Etage“ bezeichnet. Hier werden Häftlinge mit den schwersten psychischen Erkrankungen \
    eingesperrt, bis sie bereit sind, vor Gericht zu erscheinen. Am häufigsten werden sie wegen Drogendelikten oder Körperverletzung angeklagt – Anklagen, die laut Richter Steven Leifman in der Regel „vermeidbare \
    Straftaten“ sind. Er sagt, die Festnahmen seien oft das Ergebnis von Konfrontationen mit der Polizei. Geisteskranke tun oft nicht, was man ihnen sagt, wenn die Polizei am Tatort eintrifft – Konfrontationen \
    scheinen ihre Krankheit zu verschlimmern, sie werden paranoider, wahnhafter und befolgen Anweisungen weniger, so Leifman. So landen sie schwer psychisch gestört im neunten Stock, bekommen aber keine wirkliche \
    Hilfe, weil sie im Gefängnis sind. Wir haben mit Leifman das Gefängnis besichtigt. Er ist in Miami als Anwalt für Gerechtigkeit und Geisteskranke bekannt. Obwohl wir von den Wärtern nicht gerade mit offenen \
    Armen empfangen wurden, bekamen wir die Erlaubnis, Videos aufzunehmen und uns durch das Stockwerk zu bewegen. Gehen Sie in das „vergessene Stockwerk“ » . Zunächst ist es schwer zu erkennen, wo die \
    Leute sind. Die Gefangenen tragen ärmellose Gewänder. Stellen Sie sich vor, man schneidet Löcher für Arme und Füße in einen schweren Wollschlafsack – so ungefähr sehen sie aus. Sie sollen verhindern, \
    dass sich die Geisteskranken verletzen. Aus diesem Grund haben sie auch keine Schuhe, Schnürsenkel oder Matratzen. Leifman sagt, dass etwa ein Drittel aller Menschen in den Gefängnissen des Bezirks \
    Miami-Dade psychisch krank sind. Er sagt, dass die schiere Menge das System überfordert, und das Ergebnis ist das, was wir im neunten Stock sehen. Natürlich ist es ein Gefängnis, also sollte es nicht \
    warm und gemütlich sein, aber die Lichter sind grell, die Zellen sind winzig und es ist laut. Wir sehen zwei, manchmal drei Männer – manchmal in Roben, manchmal nackt, liegen oder sitzen in ihren Zellen. \
    „Ich bin der Sohn des Präsidenten. Sie müssen mich hier rausholen!“, schreit mich einer der Männer an. Er meint es absolut ernst und ist überzeugt, dass Hilfe unterwegs ist – wenn er nur das Weiße Haus \
    erreichen könnte. Leifman erzählt mir, dass diese Gefangenen-Patienten oft durch das System zirkulieren, gelegentlich in einer psychiatrischen Klinik stabilisiert werden, nur um dann ins Gefängnis \
    zurückzukehren, um sich ihren Anklagen zu stellen. In seinen Augen ist das brutal ungerecht, und er ist zu einem starken Befürworter einer Änderung der Dinge in Miami geworden. Beim Essen sprechen wir \
    später darüber, wie es dazu kam, dass Geisteskranke so weit kamen. Leifman sagt, vor 200 Jahren galten Menschen als „Irrsinnige“ und wurden ins Gefängnis gesperrt, auch wenn sie nicht angeklagt waren. \
    Sie galten einfach als unfähig, am Leben in der Gesellschaft teilzunehmen. Im Laufe der Jahre, sagt er, gab es einen öffentlichen Aufschrei, und die Geisteskranken wurden aus den Gefängnissen in Krankenhäuser \
    verlegt. Aber Leifman sagt, viele dieser psychiatrischen Kliniken waren so schrecklich, dass sie geschlossen wurden. Wohin gingen die Patienten? Nirgendwohin. Auf die Straße. In vielen Fällen wurden sie \
    obdachlos, sagt er. Sie wurden nie behandelt. Leifman sagt, 1955 befanden sich mehr als eine halbe Million Menschen in staatlichen psychiatrischen Kliniken, und heute sei diese Zahl um 90 Prozent gesunken, \
    und 40.000 bis 50.000 Menschen seien in psychiatrischen Kliniken. Der Richter sagt, er arbeite daran, dies zu ändern. Ab 2008 werden viele Häftlinge, die sonst auf die „vergessene Etage“ gebracht worden wären, ß
stattdessen in eine neue psychiatrische Einrichtung geschickt – der erste Schritt auf dem Weg zu einer langfristigen Behandlung, nicht nur zu einer Bestrafung. Leifman sagt, das sei nicht die vollständige Lösung, \
aber immerhin ein Anfang. Das Beste daran sei, dass es eine Win-Win-Lösung sei, sagt Leifman. Die Patienten gewinnen, die Familien sind erleichtert und der Staat spart Geld, indem er diese Häftlinge einfach nicht \
immer wieder durch die Einrichtung schleust. Und für Leifman ist Gerechtigkeit geschehen. E-Mail an einen Freund

"""
},

{    "Article" : """MINNEAPOLIS, Minnesota (CNN) -- Drivers who were on the Minneapolis bridge when it collapsed told harrowing tales of survival. "The whole bridge from one side of the Mississippi to the other \
    just completely gave way, fell all the way down," survivor Gary Babineau told CNN. "I probably had a 30-, 35-foot free fall. And there's cars in the water, there's cars on fire. The whole bridge is down." He \
    said his back was injured but he determined he could move around. "I realized there was a school bus right next to me, and me and a couple of other guys went over and started lifting the kids off the bridge. \
    They were yelling, screaming, bleeding. I think there were some broken bones."  Watch a driver describe his narrow escape » . At home when he heard about the disaster, Dr. John Hink, an emergency room physician, \
    jumped into his car and rushed to the scene in 15 minutes. He arrived at the south side of the bridge, stood on the riverbank and saw dozens of people lying dazed on an expansive deck. They were in the middle \
    of the Mississippi River, which was churning fast, and he had no way of getting to them. He went to the north side, where there was easier access to people. Ambulances were also having a hard time driving down \
    to the river to get closer to the scene. Working feverishly, volunteers, EMTs and other officials managed to get 55 people into ambulances in less than two hours. Occasionally, a pickup truck with a medic inside \
    would drive to get an injured person and bring him back up even ground, Hink told CNN. The rescue effort was controlled and organized, he said; the opposite of the lightning-quick collapse. "I could see the whole \
    bridge as it was going down, as it was falling," Babineau said. "It just gave a rumble real quick, and it all just gave way, and it just fell completely, all the way to the ground. And there was dust everywhere \
    and it was just like everyone has been saying: It was just like out of the movies." Babineau said the rear of his pickup truck was dangling over the edge of a broken-off section of the bridge. He said several \
    vehicles slid past him into the water. "I stayed in my car for one or two seconds. I saw a couple cars fall," he said. "So I stayed in my car until the cars quit falling for a second, then I got out real quick, \
    ran in front of my truck -- because behind my truck was just a hole -- and I helped a woman off of the bridge with me. "I just wanted off the bridge, and then I ran over to the school bus. I started grabbing \
    kids and handing them down. It was just complete chaos." He said most of the children were crying or screaming. He and other rescuers set them on the ground and told them to run to the river bank, but a few \
    needed to be carried because of their injuries.  See rescuers clamber over rubble » . Babineau said he had no rescue training. "I just knew what I had to do at the moment." Melissa Hughes, 32, of Minneapolis, \
    told The Associated Press that she was driving home when the western edge of the bridge collapsed under her. "You know that free-fall feeling? I felt that twice," Hughes said. A pickup landed on top of her car, \
    but she was not hurt. "I had no idea there was a vehicle on my car," she told AP. "It's really very surreal." Babineau told the Minneapolis Star-Tribune: "On the way down, I thought I was dead. I literally thought \
    I was dead. "My truck was completely face down, pointed toward the ground, and my truck got ripped in half. It was folded in half, and I can't believe I'm alive."  See and hear eyewitness accounts » . Bernie \
    Toivonen told CNN's "American Morning" that his vehicle was on a part of the bridge that ended up tilted at a 45-degree angle. "I knew the deck was going down, there was no question about it, and I thought I \
    was going to die," he said. After the bridge settled and his car remained upright, "I just put in park, turned the key off and said, 'Oh, I'm alive,' " he said. E-mail to a friend.
""",

    "Translation": """MINNEAPOLIS, Minnesota (CNN) -- Autofahrer, die sich auf der Minneapolis-Brücke befanden, als diese einstürzte, erzählten erschütternde Geschichten von ihrem Überleben. "Die ganze Brücke von \
    einer Seite des Mississippi zur anderen ist einfach komplett nachgegeben, komplett eingestürzt", sagte der Überlebende Gary Babineau gegenüber CNN. "Ich hatte wahrscheinlich einen freien Fall von 9, 10 Metern. \
    Und Autos sind im Wasser, Autos stehen in Flammen. Die ganze Brücke ist eingestürzt." Er sagte, sein Rücken sei verletzt, aber er sei entschlossen, sich bewegen zu können. "Mir wurde klar, dass direkt neben mir \
    ein Schulbus stand, und ich und ein paar andere Jungs gingen hinüber und begannen, die Kinder von der Brücke zu heben. Sie schrien, kreischten und bluteten. Ich glaube, sie hatten Knochenbrüche." Sehen Sie sich \
    an, wie ein Autofahrer seine knappe Rettung beschreibt » . Als er zu Hause von der Katastrophe hörte, sprang Dr. John Hink, ein Notarzt, in sein Auto und eilte innerhalb von 15 Minuten zum Unfallort. Er erreichte \
    die Südseite der Brücke, stand am Flussufer und sah Dutzende von Menschen, die benommen auf einem weitläufigen Deck lagen. Sie befanden sich mitten im schnell reißenden Mississippi, und er hatte keine Möglichkeit, \
    zu ihnen zu gelangen. Er ging auf die Nordseite, wo er leichteren Zugang zu den Menschen hatte. Auch Krankenwagen hatten Schwierigkeiten, zum Fluss hinunterzufahren, um näher an die Unfallstelle zu gelangen. \
    In fieberhafter Arbeit gelang es Freiwilligen, Rettungssanitätern und anderen Beamten, in weniger als zwei Stunden 55 Menschen in die Krankenwagen zu bringen. Gelegentlich fuhr ein Pickup mit einem Sanitäter \
    vor, um einen Verletzten zu holen und ihn wieder auf ebenem Boden zu bringen, sagte Hink gegenüber CNN. Die Rettungsaktion sei kontrolliert und organisiert gewesen, sagte er; das Gegenteil des blitzschnellen \
    Einsturzes. „Ich konnte sehen, wie die ganze Brücke einstürzte, wie sie fiel“, sagte Babineau. "Es gab nur ganz kurz ein Grollen von sich, dann gab alles nach und fiel komplett zu Boden. Überall war Staub und \
    es war, wie alle sagen: Es war wie im Film." Babineau sagte, das Heck seines Pickups baumelte über der Kante eines abgebrochenen Brückenteils. Er sagte, mehrere Fahrzeuge seien an ihm vorbei ins Wasser gerutscht. \
    "Ich blieb ein oder zwei Sekunden in meinem Auto. Ich sah ein paar Autos fallen", sagte er. "Also blieb ich in meinem Auto, bis die Autos für eine Sekunde aufhörten zu fallen, dann stieg ich ganz schnell aus, \
    rannte vor meinen Truck - denn hinter meinem Truck war nur ein Loch - und half einer Frau, mit mir von der Brücke zu kommen. "Ich wollte einfach nur von der Brücke runter und dann rannte ich zum Schulbus. \
    Ich fing an, Kinder zu packen und sie herunterzureichen. Es herrschte das totale Chaos." Er sagte, die meisten Kinder hätten geweint oder geschrien. Er und andere Rettungskräfte legten sie auf den Boden \
    und forderten sie auf, zum Flussufer zu rennen, aber einige mussten wegen ihrer Verletzungen getragen werden. Siehe Rettungskräfte klettern über Trümmer » . Babineau sagte, er habe keine Rettungsausbildung. \
    „Ich wusste einfach, was ich in dem Moment tun musste.“ Melissa Hughes, 32, aus Minneapolis, sagte der Associated Press, sie sei nach Hause gefahren, als die westliche Kante der Brücke unter ihr zusammenbrach. \
    „Kennen Sie dieses Gefühl des freien Falls? Ich habe das zweimal gespürt“, sagte Hughes. Ein Pickup landete auf ihrem Auto, aber sie wurde nicht verletzt. „Ich hatte keine Ahnung, dass ein Fahrzeug auf meinem \
    Auto war“, sagte sie der AP. „Es ist wirklich sehr surreal.“ Babineau sagte der Minneapolis Star-Tribune: „Auf dem Weg nach unten dachte ich, ich wäre tot. Ich dachte buchstäblich, ich wäre tot. \
    "Mein Truck lag komplett mit der Vorderseite nach unten, zeigte zum Boden und war in zwei Hälften gerissen. Er war in zwei Hälften gefaltet und ich kann nicht glauben, dass ich noch lebe." Sehen und hören \
    Sie Augenzeugenberichte » . Bernie Toivonen sagte gegenüber CNNs "American Morning", dass sein Fahrzeug auf einem Teil der Brücke lag, der sich schließlich um 45 Grad neigte. "Ich wusste, dass die Fahrbahndecke \
    nachgeben würde, daran bestand kein Zweifel, und ich dachte, ich würde sterben", sagte er. Nachdem sich die Brücke gesenkt hatte und sein Auto aufrecht stand, "habe ich einfach den Parkmodus eingelegt, den \
    Schlüssel abgezogen und gesagt: 'Oh, ich lebe'", sagte er. E-Mail an einen Freund.
"""
},

]

few_shot_prompt = ""

for art in articles:
    few_shot_prompt += f'Translate this text from English to German: \n\n {art.get("Article")} \n\n Translation: \n\n {art.get("Translation")}  \n\n'

prompt_with_new_article = few_shot_prompt + f'Translate this text from English to German: \n\n {dataset["test"][0].get("article")} \n\n'

# generate tokens

prompt_tokens = tokenizer(prompt_with_new_article, return_tensors='tf', max_length=7000, truncation=True, padding=True)

# get outputs

outputs = model.generate(prompt_tokens["input_ids"], max_length=7000)

# decode and print

print(f'Translated_test_article: {tokenizer.decode(outputs[0], skip_special_tokens=True)}')
