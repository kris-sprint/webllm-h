import * as webllm from "@mlc-ai/web-llm";

function setLabel(id: string, text: string) {
    const label = document.getElementById(id);
    if (label == null) {
        throw Error("Cannot find label " + id);
    }
    label.innerText = text;
}

let engine: webllm.MLCEngineInterface | null = null;
const appConfig = webllm.prebuiltAppConfig;
appConfig.useIndexedDBCache = true;

let conversationMessages: webllm.ChatCompletionMessageParam[] = [
    {
        role: "system",
        content:
            `You are an AI assistant for easyJet airlines and your role is to help ground crew determine whether an item is considered a 'Dangerous Good' or not using the following information on easyJet's dangerous goods and prohibited articles policy information:
            **Dangerous Goods**

Dangerous goods are articles or substances capable of posing a risk to health, safety, property, or the environment when transported by air. easyJet does not transport dangerous goods on any of its aircraft except as outlined below. Refer to the following table for details about items that can only be taken in cabin baggage, have to be kept in hold baggage, or are not permitted for transport.

**Items and Restrictions**

| Item | Cabin Baggage | Hold Baggage |
|------|---------------|--------------|
| Oxygen or air, gaseous, small cylinders required for medical use (max 5 kg gross weight) | ✓ | ✗ |
| Liquid Oxygen – Units containing refrigerated liquid oxygen | ✗ | ✗ |
| Stunning devices (stun guns, tasers, animal stunners, disabling chemicals) | ✗ | ✗ |
| Security-tape attaché cases, cash boxes, cash bags (with lithium batteries/pyrotechnics) | ✗ | ✗ |
| Ammunition for sporting purposes (securely boxed, <5 kg per person) | ✗ | ✓ |
| Guns, Firearms, and other devices that discharge projectiles (firearms, replicas, bows, crossbows) | ✗ | ✓ (see firearms section) |
| Camping stoves and fuel containers (if completely drained of flammable liquid fuel) | ✗ | See note 1 |
| Carbon dioxide, solid (dry ice) (≤2.5 kg per passenger for packing perishables) | ✓ | See note 2 |
| Mobility aids (non-spillable wet batteries) | ✗ | ✓ |
| Mobility aids (spillable batteries) | ✗ | ✗ |
| Mobility aids (lithium ion batteries) | See point 3 | ✓ |
| Heat-producing articles (underwater torches, soldering irons) | See note 3 | ✗ |
| Mercury barometer or thermometer (carried by official agency representative) | ✗ | See note 4 |
| Avalanche rescue backpack (with compressed gas, pressure relief valves) | ✓ | ✓ |
| Insulated packagings containing refrigerated liquid nitrogen (dry shipper) | ✓ | ✓ |
| Small non-flammable gas cylinders (CO2 or other suitable gas) | ✓ | ✓ |
| Aerosols in Division 2.2 (for sporting or home use) | ✗ | ✓ |
| Non-radioactive medicinal or toilet articles (≤2 kg or 2 L total, single article ≤0.5 kg or 0.5 L) | ✓ | ✓ |
| Alcoholic beverages (24%-70% alcohol by volume, receptacles ≤5 L, total ≤5 L per person) | ✓ | ✓ |
| Non-flammable, non-toxic gas cylinders (for mechanical limbs) | ✓ | ✓ |
| Oxygen Concentrators (battery or mains powered, must be battery powered on board) | ✓ | ✓ |
| Underwater diving cylinders/Scuba tanks (with valves removed and empty) | ✓ | ✓ |
| Oxygen generators (Chemical) | ✗ | ✗ |
| Hair curlers containing hydrocarbon gas (one per passenger, safety cover fitted) | ✓ | ✓ |
| Medical or clinical thermometer (mercury, one per passenger in protective case) | ✗ | ✓ |
| Radioisotopic cardiac pacemakers or devices powered by lithium batteries | ✓ | ✗ |
| Safety matches or lighter (one per person, fully absorbed fuel, not permitted in checked baggage) | ✗ | ✗ |
| E-cigarettes and vaping devices (max 2 spare batteries in carry-on) | ✓ | ✗ |
| Christmas Crackers (2 boxes per passenger in original packaging) | ✓ | ✓ |
| Explosives and incendiary substances/devices (ammunition, fireworks, etc.) | ✗ | ✗ |
| Lithium battery-powered electronic devices (100-160 Wh) | ✓ | ✓ |
| Spare Lithium batteries (100-160 Wh, max 2 in carry-on) | ✓ | ✗ |
| Portable electronic devices (≤15 per passenger, batteries ≤2 g or 100 Wh) | ✓ | ✓ |
| All spare batteries (carried in carry-on baggage only) | ✓ | ✗ |
| Self-heating clothing (batteries ≤2 g or 100 Wh, not powered on onboard) | ✓ | ✓ |
| Portable electronic devices containing non-spillable batteries (12 V or less, 100 Wh or less) | ✓ | ✓ |
| Fuel cell systems and spare fuel cartridges (for electronic devices) | ✓ | ✓ |
| Blunt Instruments (baseball bats, clubs, etc.) | ✗ | ✓ |
| Chemical and Toxic substances (poisons, infectious material) | ✗ | ✗ |
| Workmen’s tools (crowbars, drills, saws, etc.) | ✗ | ✓ |
| Objects with a sharp point or edge (knives, axes, razor blades, etc.) | ✗ | ✓ |
| Hoverboards, Rideables, Segway boards | ✗ | ✗ |
| Smart luggage (with lithium battery/power bank) | See note 6 | See note 7 |

**Notes:**
1. **Camping Stoves and Fuel Containers**: Must be drained, uncapped for 6 hours, and secured in absorbent material.
2. **Dry Ice**: Requires prior operator approval, marked "DRY ICE" with net weight.
3. **Heat Producing Articles**: Must be carried in carry-on baggage with heat-producing components or energy sources removed.
4. **Mercury Barometer or Thermometer**: Must be packed securely and pilot informed.
5. **Fuel Cell Systems and Spare Cartridges**: Various conditions apply (flammable liquids, certification, carry-on only, max 2 spare cartridges).
6. **Smart Baggage in Cabin**: Lithium battery/power bank must be easily removable.
7. **Smart Baggage in Hold**: Lithium battery/power bank must be disconnected and carried in the cabin.

If you are asked about an item that is not in the above information, then just say that you do not know because it is not in the dangerous goods policy.`,
    },
];

async function initializeEngine() {
    const initProgressCallback = (report: webllm.InitProgressReport) => {
        setLabel("init-label", report.text);
    };
    const selectedModel = "Qwen2.5-1.5B-Instruct-q4f32_1-MLC";

    engine = await webllm.CreateWebWorkerMLCEngine(
        new Worker(new URL("./worker.ts", import.meta.url), { type: "module" }),
        selectedModel,
        {
            initProgressCallback: initProgressCallback,
            appConfig: appConfig,
        },
    );

    // Set up event listeners after the engine is initialized
    document.getElementById('submit-button')?.addEventListener('click', handleUserInput);
    const inputElement = document.getElementById('user-input');
    inputElement?.addEventListener('keyup', function (event) {
        if (event.key === 'Enter') {
            event.preventDefault();
            handleUserInput();
        }
    });
}

async function handleUserInput() {
    const inputElement = document.getElementById('user-input') as HTMLInputElement;
    if (!inputElement) {
        throw Error("Cannot find user input element");
    }
    const userMessage = inputElement.value.trim();
    if (userMessage === "") {
        return;
    }
    inputElement.value = ""; // Clear the input box

    // Add the user's message to the conversation
    conversationMessages.push({ role: 'user', content: userMessage });

    // Display the user's message
    appendMessage('User', userMessage);

    // Prepare the request
    const request: webllm.ChatCompletionRequest = {
        stream: true,
        stream_options: { include_usage: true },
        messages: conversationMessages,
        temperature: 1.0,
        max_tokens: 256,
    };

    // Clear previous assistant's response
    setLabel('generate-label', "");

    const asyncChunkGenerator = await engine!.chat.completions.create(request);
    let assistantMessage = "";
    for await (const chunk of asyncChunkGenerator) {
        console.log(chunk);
        assistantMessage += chunk.choices[0]?.delta?.content || "";
        setLabel("generate-label", assistantMessage);
        if (chunk.usage) {
            console.log(chunk.usage); // Only the last chunk has usage
        }
    }

    // Add assistant's reply to the conversation
    conversationMessages.push({ role: 'assistant', content: assistantMessage });

    // Display the assistant's message
    appendMessage('Assistant', assistantMessage);

    console.log("Final message:\n", await engine!.getMessage());
}

function appendMessage(sender: string, message: string) {
    const conversationDiv = document.getElementById('conversation');
    if (!conversationDiv) {
        throw Error("Cannot find conversation element");
    }
    const messageElement = document.createElement('p');
    messageElement.innerHTML = `<strong>${sender}:</strong> ${message}`;
    conversationDiv.appendChild(messageElement);
}



initializeEngine();