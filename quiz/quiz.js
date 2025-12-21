const yay_emoji = [
    "\u{1F44F}",
    "\u{1F601}",
    "\u{1F60A}",
    "\u{1F913}",
    "\u{1F920}",
];

const sad_emoji = [
    "\u{1F61F}",
    "\u{1F626}",
    "\u{1F62D}",
    "\u{1F922}",
    "\u{1F92E}",
];

var yay_emoji_idx = 0;
var sad_emoji_idx = 0;

function next_yay () {
    yay_emoji_idx = (yay_emoji_idx + 1) % yay_emoji.length;
    return yay_emoji[yay_emoji_idx];
};

function next_sad () {
    sad_emoji_idx = (sad_emoji_idx + 1) % sad_emoji.length;
    return sad_emoji[sad_emoji_idx];
};


async function hash_digest (message) {
    const encoder = new TextEncoder();
    const data = encoder.encode(message); 
    const buffer = await crypto.subtle.digest("SHA-256", data); 
    const explode = Array.from(new Uint8Array(buffer)); 
    const hash = explode.map(byte => byte.toString(16).padStart(2, "0")).join(""); 

    return hash;
}


async function score_quiz () {
    for (var i = 0; i < document.forms.length; i++) {
	const form = document.forms[i];
	var score = 0;

	const result = document.getElementById("result");
	result.innerHTML = "";

	Array.from(form.elements).forEach((input) => {
            if (input.type == "radio") {
		const multi = document.getElementById(input.name);

		// clear any previous annotations
		input.labels[0].style.color = "black";
		input.labels[0].style.fontWeight = "normal";
		multi.innerHTML = "";

    		if (input.checked) {
    		    const message = input.name.concat(input.id);

		    hash_digest(message).then(
			(hash) => {
			    var emoji = "";

			    if (answer_key.includes(hash)) {
				emoji = next_yay();
				input.labels[0].style.fontWeight = "bold";

				score += 1;
			    } else {
				emoji = next_sad();
				input.labels[0].style.color = "red";
			    };


			    const grade = [
				Math.round(score / num_questions * 100.0).toString(),
				" / 100",
			    ]

			    result.innerHTML = grade.join("");

			    // annotate the answer or decoy
			    const decoy = decoys.get(input.name);
			    var cite = "";

			    if (decoy[input.id].cite) {
				const link = [
				    "&nbsp;&nbsp;<a",
				    ' target="_blank" href="',
				    decoy[input.id].cite,
				    '">(ref)</a>',
				];

				cite = link.join("");
			    };

			    const para = [
				"<p><span style='font-size:2em;'>",
				emoji,
				"</span> <strong>",
				decoy[input.id].info,
				"</strong>",
				cite,
				"</p>",
			    ];

			    multi.innerHTML = para.join("");
			}
		    );
		};
            };
	});
    };
}
