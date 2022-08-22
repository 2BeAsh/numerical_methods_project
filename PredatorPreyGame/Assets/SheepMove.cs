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

        List<GameObject> preyList = new();
        foreach (GameObject fooObj in GameObject.FindGameObjectsWithTag("Prey"))  //Maybe find a way to only run this code if an object is removed
        {
            preyList.Add(fooObj);
        }
        int numberOfPrey = preyList.Count;
        foreach (GameObject prey in preyList)
        {
            Vector3 velocity = new Vector3(0, 0, 0); // Velocity of j'th particle, will be updated
            foreach (GameObject otherPrey in preyList) // The k'th particle
            {
                if (otherPrey == prey)
                {
                    continue;
                }

                float dist = Vector3.Distance(prey.transform.position, otherPrey.transform.position);
                Vector3 velocityShortRange = (prey.transform.position - otherPrey.transform.position) / (dist * dist);
                Vector3 velocityLongRange = longRangeAttraction * (prey.transform.position - otherPrey.transform.position);

                velocity += (velocityShortRange - velocityLongRange) / numberOfPrey;
                
            }
            float dist_pred = Vector3.Distance(prey.transform.position, playerObj.transform.position);
            Vector3 velocityPredRepulsion = predatorRepulsion * (prey.transform.position - playerObj.transform.position) / (dist_pred * dist_pred);
            velocity += velocityPredRepulsion;

            // Update position from velocity calculations
            //prey.transform.position += velocity * Time.fixedDeltaTime;  - Doesn't allow for physics
            Rigidbody2D preyRB = prey.GetComponent<Rigidbody2D>(); // Get j'th particle's rigidbody and update its position
           preyRB.MovePosition(preyRB.position + ((Vector2)velocity * Time.fixedDeltaTime)); // Changed to fixedDeltaTime since using FixedUpdate for rigid body
            
        }

    }

}
