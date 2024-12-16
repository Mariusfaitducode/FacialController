import asyncio
import websockets
import json
from datetime import datetime

async def test_client():
    uri = "ws://localhost:8765"
    try:
        async with websockets.connect(uri) as websocket:
            print(f"Connecté au serveur WebSocket sur {uri}")
            print("En attente des données... (Ctrl+C pour quitter)")
            
            while True:
                try:
                    message = await websocket.recv()
                    print("\nMessage reçu brut:", message)  # Debug log
                    data = json.loads(message)
                    
                    # Formatage plus lisible des données
                    timestamp = datetime.fromtimestamp(data['timestamp']).strftime('%H:%M:%S.%f')[:-3]
                    face_data = data['data']
                    
                    print("\n" + "="*50)
                    print(f"Timestamp: {timestamp}")
                    print(f"Face ID: {face_data['face_id']}")
                    print(f"👁  Clignement: {'OUI' if face_data['blink_detected'] else 'NON'}")
                    print(f"👄 Bouche: {'OUVERTE' if face_data['mouth_open'] else 'FERMÉE'}")
                    print(f"📊 Ratio bouche: {face_data['mouth_ratio']:.3f}")
                    print(f"🔢 Total clignements: {face_data['total_blinks']}")
                except Exception as e:
                    print(f"Erreur lors de la réception: {e}")

    except TimeoutError:
        print("❌ Impossible de se connecter au serveur WebSocket (timeout)")
        print("   Assurez-vous que le programme principal est en cours d'exécution")
    except ConnectionRefusedError:
        print("❌ Connexion refusée par le serveur WebSocket")
        print("   Assurez-vous que le programme principal est en cours d'exécution")
    except KeyboardInterrupt:
        print("\n✨ Arrêt du client WebSocket")
    except Exception as e:
        print(f"❌ Erreur inattendue: {str(e)}")

if __name__ == "__main__":
    asyncio.run(test_client()) 