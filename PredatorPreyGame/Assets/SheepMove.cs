using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class SheepMove : MonoBehaviour
{
    public float longRangeAttraction;
    public float predatorRepulsion;
    private GameObject playerObj = null;

    // Start is called before the first frame update
    private void Start()
    {
        
        if (playerObj == null)
        {
            playerObj = GameObject.FindGameObjectWithTag("Player");
        }
    }


    private void FixedUpdate()
    {
        /* Movement 
        Loop through each member and calculate their velocity, and from that update their position.
        Calculation requires looping through all members again, excluding the first. */

        // Find all current prey and add them to the list preyList
        List<GameObject> preyList = new();
        foreach (GameObject fooObj in GameObject.FindGameObjectsWithTag("Prey"))  //Maybe find a way to only run this code if an object is removed
        {
            preyList.Add(fooObj);
        }
        // Loop through each prey in preyList and update its position by calculating its velocity
        foreach (GameObject prey in preyList)
        {
            Vector2 velocity = Move_prey(preyList, prey);

            // Align sprite direction with velocity
            SpriteRenderer spriteRenderer = prey.GetComponent<SpriteRenderer>();
            if (velocity.x < 0)
            {
                spriteRenderer.flipX = true;
            }
            else if (velocity.x > 0)
            {
                spriteRenderer.flipX = false;
            }

            // Movement
            Rigidbody2D preyRB = prey.GetComponent<Rigidbody2D>(); // Get j'th particle's rigidbody and update its position
            preyRB.MovePosition(preyRB.position + (velocity * Time.fixedDeltaTime)); // Changed to fixedDeltaTime since using FixedUpdate for rigid body

        }

    }

    private Vector2 Move_prey(List<GameObject> preyList, GameObject prey)
    {       int numberOfPrey = preyList.Count;
            Vector2 velocity = new Vector2(0, 0); // Velocity of j'th particle, will be updated
            foreach (GameObject otherPrey in preyList) // The k'th particle
            {
                if (otherPrey == prey)
                {
                    continue;
                }

                float dist = Vector2.Distance(prey.transform.position, otherPrey.transform.position);
                Vector2 velocityShortRange = (prey.transform.position - otherPrey.transform.position) / (dist * dist);
                Vector2 velocityLongRange = longRangeAttraction * (prey.transform.position - otherPrey.transform.position);

                velocity += (velocityShortRange - velocityLongRange) / numberOfPrey;

            }
            float dist_pred = Vector2.Distance(prey.transform.position, playerObj.transform.position);
            Vector2 velocityPredRepulsion = predatorRepulsion * (prey.transform.position - playerObj.transform.position) / Mathf.Pow(dist_pred, 2);
            velocity += velocityPredRepulsion;
        return velocity;
    }

}
