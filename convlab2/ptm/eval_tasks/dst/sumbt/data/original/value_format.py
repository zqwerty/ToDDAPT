def formatting(d, s, v):
    if v == "cambridge contemporary art museum":
        v = "cambridge contemporary art"
    if v == "cafe jello museum":
        v = "cafe jello gallery"
    if v == "whippple museum":
        v = "whipple museum of the history of science"
    if v == "st christs college":
        v = "christ s college"
    if v == "abc theatre":
        v = "adc theatre"
    if d == "train" and v == "london":
        v = "london kings cross"
    if v == "the castle galleries":
        v = "castle galleries"
    if v == "cafe jello":
        v = "cafe jello gallery"
    if v == "cafe uno":
        v = "caffe uno"
    if v == "el shaddia guesthouse":
        v = "el shaddai"
    if v == "kings college":
        v = "king s college"
    if v == "saint johns college":
        v = "saint john s college"
    if v == "kettles yard":
        v = "kettle s yard"
    if v == "grafton hotel":
        v = "grafton hotel restaurant"
    if v == "churchills college":
        v = "churchill college"
    if v == "the churchill college":
        v = "churchill college"
    if v == "portugese":
        v = "portuguese"
    if v == "lensfield hotel":
        v = "the lensfield hotel"
    if v == "rosas bed and breakfast":
        v = "rosa s bed and breakfast"
    if v == "pizza hut fenditton":
        v = "pizza hut fen ditton"
    if v == "great saint marys church":
        v = "great saint mary s church"
    if v == "alimentum":
        v = "restaurant alimentum"
    if v == "cow pizza kitchen and bar":
        v = "the cow pizza kitchen and bar"
    if v == "shiraz":
        v = "shiraz restaurant"
    if v == "cherry hinton village centre":
        v = "the cherry hinton village centre"
    if v == "christ college":
        v = "christ s college"
    if v == "peoples portraits exhibition at girton college":
        v = "people s portraits exhibition at girton college"
    if v == "saint catharines college":
        v = "saint catharine s college"
    if v == "the maharajah tandoor":
        v = "maharajah tandoori restaurant"
    if v == "efes":
        v = "efes restaurant"
    if v == "the gonvile hotel":
        v = "gonville hotel"
    if v == "abbey pool":
        v = "abbey pool and astroturf pitch"
    if v == "the cambridge arts theatre":
        v = "cambridge arts theatre"
    if v == "sheeps green and lammas land park fen causeway":
        v = "sheep s green and lammas land park fen causeway"
    if v == "lensfield hotel":
        v = "the lensfield hotel"
    if v == "rosas bed and breakfast":
        v = "rosa s bed and breakfast"
    if v == "little saint marys church":
        v = "little saint mary s church"
    if v == "cambridge punter":
        v = "the cambridge punter"
    if v == "pizza hut":
        v = "pizza hut city centre"
    if v == "good luck":
        v = "the good luck chinese food takeaway"
    if v == "lucky star":
        v = "the lucky star"
    if v == "cambridge contemporary art museum":
        v = "cambridge contemporary art"
    if v == "cow pizza kitchen and bar":
        v = "the cow pizza kitchen and bar"
    if v == "river bar steakhouse and grill":
        v = "the river bar steakhouse and grill"
    if v == "chiquito":
        v = "chiquito restaurant bar"
    if v == "king hedges learner pool":
        v = "kings hedges learner pool"
    if v == "dontcare":
        v = "dontcare"
    if v == "does not care":
        v = "dontcare"
    if v == "corsican":
        v = "corsica"
    if v == "barbeque":
        v = "barbecue"
    if v == "center":
        v = "centre"
    if v == "east side":
        v = "east"
    if s == "pricerange":
        s = "pricerange"
    if s == "pricerange" and v == "mode":
        v = "moderate"
    if v == "not mentioned":
        v = ""
    if v == "thai and chinese":  # only one such type, throw away
        v = "chinese"
    if s == "area" and v == "n":
        v = "north"
    if s == "pricerange" and v == "ch":
        v = "cheap"
    if v == "moderate -ly":
        v = "moderate"
    if s == "area" and v == "city center":
        v = "centre"
    if s == "food" and v == "sushi":  # sushi only appear once in the training dataset. doesnt matter throw it away or not
        v = "japanese"
    if v == "oak bistro":
        v = "the oak bistro"
    if v == "golden curry":
        v = "the golden curry"
    if v == "meze bar restaurant":
        v = "meze bar"
    if v == "golden house golden house":
        v = "golden house"
    if v == "missing sock":
        v = "the missing sock"
    if v == "the yippee noodle bar":
        v = "yippee noodle bar"
    if v == "fitzbillies":
        v = "fitzbillies restaurant"
    if v == "slug and lettuce":
        v = "the slug and lettuce"
    if v == "copper kettle":
        v = "the copper kettle"
    if v == "city stop":
        v = "city stop restaurant"
    if v == "cambridge lodge":
        v = "cambridge lodge restaurant"
    if v == "ian hong house":
        v = "lan hong house"
    if v == "lan hong":
        v = "lan hong house"
    if v == "hotpot":
        v = "the hotpot"
    if v == "the dojo noodle bar":
        v = "dojo noodle bar"
    if v == "cambridge chop house":
        v = "the cambridge chop house"
    if v == "nirala":
        v = "the nirala"
    if v == "gardenia":
        v = "the gardenia"
    if v == "the americas":
        v = "americas"
    if v == "guest house":
        v = "guesthouse"
    if v == "margherita":
        v = "la margherita"
    if v == "gonville":
        v = "gonville hotel"
    if s == "parking" and v == "free":
        v = "yes"
    if d == "hotel" and s == "name":
        if v == "acorn" or v == "acorn house":
            v = "acorn guest house"
        if v == "cambridge belfry":
            v = "the cambridge belfry"
        if v == "huntingdon hotel":
            v = "huntingdon marriott hotel"
        if v == "alexander":
            v = "alexander bed and breakfast"
        if v == "lensfield hotel":
            v = "the lensfield hotel"
        if v == "university arms":
            v = "university arms hotel"
        if v == "city roomz":
            v = "cityroomz"
        if v == "ashley":
            v = "ashley hotel"
    if d == "train":
        if s == "destination" or s == "departure":
            if v == "bishop stortford":
                v = "bishops stortford"
            if v == "bishops storford":
                v = "bishops stortford"
            if v == "birmingham":
                v = "birmingham new street"
            if v == "stansted":
                v = "stansted airport"
            if v == "leicaster":
                v = "leicester"
    if d == "attraction":
        if v == "cambridge temporary art":
            v = "contemporary art museum"
        if v == "cafe jello":
            v = "cafe jello gallery"
        if v == "fitzwilliam" or v == "fitzwilliam museum":
            v = "the fitzwilliam museum"
        if v == "contemporary art museum":
            v = "cambridge contemporary art"
        if v == "man on the moon":
            v = "the man on the moon"
        if v == "christ college":
            v = "christ s college"
        if v == "old school":
            v = "old schools"
        if v == "cambridge punter":
            v = "the cambridge punter"
        if v == "queen s college":
            v = "queens college"
        if v == "all saint s church":
            v = "all saints church"
        if v == "fez club":
            v = "the fez club"
        if v == "parkside":
            v = "parkside pools"
        if v == "saint john s college .":
            v = "saint john s college"
        if v == "the mumford theatre":
            v = "mumford theatre"
        if v == "corn cambridge exchange":
            v = "the cambridge corn exchange"
    if d == "taxi":
        if v == "london kings cross train station":
            v = "london kings cross"
        if v == "stevenage train station":
            v = "stevenage"
        if v == "junction theatre":
            v = "the junction"
        if v == "bishops stortford train station":
            v = "bishops stortford"
        if v == "cambridge train station":
            v = "cambridge"
        if v == "citiroomz":
            v = "cityroomz"
        if v == "london liverpool street train station":
            v = "london liverpool street"
        if v == "norwich train station":
            v = "norwich"
        if v == "kings college":
            v = "king s college"
        if v == "the ghandi" or v == "ghandi":
            v = "the gandhi"
        if v == "ely train station":
            v = "ely"
        if v == "stevenage train station":
            v = "stevenage"
        if v == "peterborough train station":
            v = "peterborough"
        if v == "london kings cross train station":
            v = "london kings cross"
        if v == "kings lynn train station":
            v = "kings lynn"
        if v == "stansted airport train station":
            v = "stansted airport"
        if v == "acorn house":
            v = "acorn guest house"
        if v == "queen s college":
            v = "queens college"
        if v == "leicester train station":
            v = "leicester"
        if v == "the gallery at 12":
            v = "gallery at 12 a high street"
        if v == "caffee uno":
            v = "caffe uno"
        if v == "stevenage train station":
            v = "stevenage"
        if v == "finches":
            v = "finches bed and breakfast"
        if v == "broxbourne train station":
            v = "broxbourne"
        if v == "country folk museum":
            v = "cambridge and county folk museum"
        if v == "ian hong":
            v = "lan hong house"
        if v == "the byard art museum":
            v = "byard art"
        if v == "cambridge belfry":
            v = "the cambridge belfry"
        if v == "birmingham new street train station":
            v = "birmingham new street"
        if v == "man on the moon concert hall":
            v = "the man on the moon"
        if v == "st . john s college":
            v = "saint john s college"
        if v == "st johns chop house":
            v = "saint johns chop house"
        if v == "fitzwilliam museum":
            v = "the fitzwilliam museum"
        if v == "cherry hinton village centre":
            v = "the cherry hinton village centre"
        if v == "maharajah tandoori restaurant4":
            v = "maharajah tandoori restaurant"
        if v == "the soul tree":
            v = "soul tree nightclub"
        if v == "cherry hinton village center":
            v = "the cherry hinton village centre"
        if v == "aylesbray lodge":
            v = "aylesbray lodge guest house"
        if v == "the alexander bed and breakfast":
            v = "alexander bed and breakfast"
        if v == "shiraz .":
            v = "shiraz restaurant"
        if v == "tranh binh":
            v = "thanh binh"
        if v == "riverboat georginawd":
            v = "riverboat georgina"
        if v == "lovell ldoge":
            v = "lovell lodge"
        if v == "alyesbray lodge hotel":
            v = "aylesbray lodge guest house"
        if v == "wandlebury county park":
            v = "wandlebury country park"
        if v == "the galleria":
            v = "galleria"
        if v == "cambridge artw2orks":
            v = "cambridge artworks"
    return v