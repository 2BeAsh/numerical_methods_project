using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class rayCastSheepMove : MonoBehaviour
{
    public float longRangeAttraction;
    public float predatorRepulsion;
    private GameObject playerObj = null;
    public ContactFilter2D movementFilter; // For ray casting - Finding obstacles
    public float collisionOffset = 0.05f; // For ray casting 
    public int layer;

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

        // Find all current prey
        List<GameObject> preyList = new();
        foreach (GameObject fooObj in GameObject.FindGameObjectsWithTag("Prey"))  //Maybe find a way to only run this code if an object is removed
        {
            preyList.Add(fooObj);
        }

        foreach (GameObject prey in preyList)
        {
            Vector2 velocity = Move_prey(preyList, prey);

            // Use Raycast to check if there is a collision object
            if (checkCollision(layer, prey, velocity))
            {
                continue; // If there is a collision, don't update position of this prey - Might want to make a while loop that continues until gets true
            }
            else
            {
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
                preyRB.MovePosition(preyRB.position + ((Vector2)velocity * Time.fixedDeltaTime)); // Changed to fixedDeltaTime since using FixedUpdate for rigid body

            }


            /*
            // Check for collisions
            Rigidbody2D preyRB = prey.GetComponent<Rigidbody2D>(); // Get j'th particle's rigidbody and update its position
            List<RaycastHit2D> castCollisions = new List<RaycastHit2D>();
            float velocityNorm = (velocity.x * velocity.x) + (velocity.y * velocity.y);
            int count = preyRB.Cast(
                velocity,
                movementFilter,
                castCollisions,
                velocityNorm * Time.fixedDeltaTime + collisionOffset);

            if (count == 0)
            {
                preyRB.MovePosition(preyRB.position + (Vector2)velocity * Time.fixedDeltaTime);
            }
            else if (count != 0)
            {
                Debug.Log("collision");
            }
            */
            // Update position from velocity calculations
            //prey.transform.position += velocity * Time.fixedDeltaTime;  - Doesn't allow for physics

        }

    }


    private Vector2 Move_prey(List<GameObject> preyList, GameObject prey)
    {
        int numberOfPrey = preyList.Count;
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
        return velocity;
    }


    private bool checkCollision(int layer, GameObject prey, Vector2 velocity)
    {
        int layerMask = 1 << layer;
        //layerMask = ~layerMask; // Collide against everything except object on specified layer
        RaycastHit hit;

        float velocity_norm = Mathf.Sqrt(velocity.x * velocity.x + velocity.y * velocity.y);

        if (Physics.Raycast(prey.transform.position, velocity, out hit, velocity_norm, layerMask))
        {
            Debug.Log("Raycast Hit");
            return true;
        }
        else
        {
            return false;
        }
    }

}
